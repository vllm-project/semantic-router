/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

import (
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func newFileStoreTestStore() (*FileStore, string) {
	GinkgoHelper()

	tempDir, err := os.MkdirTemp("", "filestore-test-*")
	Expect(err).NotTo(HaveOccurred())

	store, err := NewFileStore(tempDir, NewMemoryMetadataRegistry())
	Expect(err).NotTo(HaveOccurred())

	return store, tempDir
}

func cleanupFileStoreTestDir(tempDir string) {
	GinkgoHelper()
	Expect(os.RemoveAll(tempDir)).To(Succeed())
}

var _ = Describe("FileStore construction and saving", func() {
	var tempDir string

	BeforeEach(func() {
		_, tempDir = newFileStoreTestStore()
	})

	AfterEach(func() {
		cleanupFileStoreTestDir(tempDir)
	})

	Context("NewFileStore", func() {
		It("should create the files subdirectory", func() {
			info, err := os.Stat(tempDir + "/files")
			Expect(err).NotTo(HaveOccurred())
			Expect(info.IsDir()).To(BeTrue())
		})

		It("should return error for invalid path", func() {
			_, err := NewFileStore("/dev/null/invalid", NewMemoryMetadataRegistry())
			Expect(err).To(HaveOccurred())
		})

		It("should require a metadata registry", func() {
			_, err := NewFileStore(tempDir, nil)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("file metadata registry is required"))
		})
	})
})

var _ = Describe("FileStore saving", func() {
	var (
		store   *FileStore
		tempDir string
	)

	BeforeEach(func() {
		store, tempDir = newFileStoreTestStore()
	})

	AfterEach(func() {
		cleanupFileStoreTestDir(tempDir)
	})

	Context("Save", func() {
		It("should save file and return a record", func() {
			content := []byte("hello world")
			record, err := store.Save("test.txt", content, "assistants")

			Expect(err).NotTo(HaveOccurred())
			Expect(record).NotTo(BeNil())
			Expect(record.ID).To(HavePrefix("file_"))
			Expect(record.Object).To(Equal("file"))
			Expect(record.Bytes).To(Equal(int64(11)))
			Expect(record.Filename).To(Equal("test.txt"))
			Expect(record.Purpose).To(Equal("assistants"))
			Expect(record.Status).To(Equal("uploaded"))
			Expect(record.CreatedAt).To(BeNumerically(">", 0))
		})

		It("should write file to disk", func() {
			content := []byte("file content")
			record, err := store.Save("data.txt", content, "assistants")
			Expect(err).NotTo(HaveOccurred())

			// Read back from disk.
			data, err := os.ReadFile(tempDir + "/files/" + record.ID + "/data.txt")
			Expect(err).NotTo(HaveOccurred())
			Expect(data).To(Equal(content))
		})

		It("should sanitize path traversal in filename", func() {
			content := []byte("malicious content")
			record, err := store.Save("../../etc/passwd.txt", content, "assistants")

			Expect(err).NotTo(HaveOccurred())
			// Filename should be sanitized to just the base name.
			Expect(record.Filename).To(Equal("passwd.txt"))

			// File should be written inside the store directory, not at ../../etc/.
			data, err := os.ReadFile(tempDir + "/files/" + record.ID + "/passwd.txt")
			Expect(err).NotTo(HaveOccurred())
			Expect(data).To(Equal(content))

			// The traversal target should NOT exist.
			_, err = os.Stat(tempDir + "/../../etc/passwd.txt")
			Expect(os.IsNotExist(err)).To(BeTrue())
		})

		It("should reject empty filename", func() {
			_, err := store.Save("", []byte("data"), "assistants")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("invalid filename"))
		})

		It("should generate unique IDs for different saves", func() {
			r1, err := store.Save("a.txt", []byte("a"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			r2, err := store.Save("b.txt", []byte("b"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			Expect(r1.ID).NotTo(Equal(r2.ID))
		})

		It("should save file content from a reader", func() {
			content := strings.Repeat("streamed content\n", 128)
			record, err := store.SaveFromReader("streamed.txt", strings.NewReader(content), "assistants")

			Expect(err).NotTo(HaveOccurred())
			Expect(record.Bytes).To(Equal(int64(len(content))))

			data, err := store.Read(record.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(string(data)).To(Equal(content))
		})
	})
})

var _ = Describe("FileStore record access", func() {
	var (
		store   *FileStore
		tempDir string
	)

	BeforeEach(func() {
		store, tempDir = newFileStoreTestStore()
	})

	AfterEach(func() {
		cleanupFileStoreTestDir(tempDir)
	})

	Context("Read", func() {
		It("should read saved file content", func() {
			content := []byte("test content for read")
			record, err := store.Save("read.txt", content, "assistants")
			Expect(err).NotTo(HaveOccurred())

			data, err := store.Read(record.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(data).To(Equal(content))
		})

		It("should return error for non-existent file", func() {
			_, err := store.Read("file_nonexistent")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("file not found"))
		})
	})

	Context("Delete", func() {
		It("should delete file from disk and registry", func() {
			record, err := store.Save("del.txt", []byte("delete me"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			err = store.Delete(record.ID)
			Expect(err).NotTo(HaveOccurred())

			// File should no longer be readable.
			_, err = store.Read(record.ID)
			Expect(err).To(HaveOccurred())

			// Directory should be removed from disk.
			_, err = os.Stat(tempDir + "/files/" + record.ID)
			Expect(os.IsNotExist(err)).To(BeTrue())
		})

		It("should return error for non-existent file", func() {
			err := store.Delete("file_nonexistent")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("file not found"))
		})
	})
})

var _ = Describe("FileStore listing and lookup", func() {
	var (
		store   *FileStore
		tempDir string
	)

	BeforeEach(func() {
		store, tempDir = newFileStoreTestStore()
	})

	AfterEach(func() {
		cleanupFileStoreTestDir(tempDir)
	})

	Context("List", func() {
		It("should return empty list when no files saved", func() {
			records := store.List()
			Expect(records).To(BeEmpty())
		})

		It("should return all saved files", func() {
			_, err := store.Save("a.txt", []byte("a"), "assistants")
			Expect(err).NotTo(HaveOccurred())
			_, err = store.Save("b.txt", []byte("b"), "assistants")
			Expect(err).NotTo(HaveOccurred())
			_, err = store.Save("c.txt", []byte("c"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			records := store.List()
			Expect(records).To(HaveLen(3))
		})

		It("should not include deleted files", func() {
			r1, err := store.Save("a.txt", []byte("a"), "assistants")
			Expect(err).NotTo(HaveOccurred())
			_, err = store.Save("b.txt", []byte("b"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			err = store.Delete(r1.ID)
			Expect(err).NotTo(HaveOccurred())

			records := store.List()
			Expect(records).To(HaveLen(1))
		})
	})

	Context("Get", func() {
		It("should return the correct record", func() {
			saved, err := store.Save("get.txt", []byte("content"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			record, err := store.Get(saved.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(record.ID).To(Equal(saved.ID))
			Expect(record.Filename).To(Equal("get.txt"))
			Expect(record.Purpose).To(Equal("assistants"))
		})

		It("should return error for non-existent file", func() {
			_, err := store.Get("file_nonexistent")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("file not found"))
		})

		It("should return defensive record copies", func() {
			saved, err := store.Save("copy.txt", []byte("content"), "assistants")
			Expect(err).NotTo(HaveOccurred())

			record, err := store.Get(saved.ID)
			Expect(err).NotTo(HaveOccurred())
			record.Filename = "mutated.txt"

			records := store.List()
			Expect(records).To(HaveLen(1))
			records[0].Purpose = "mutated"

			reloaded, err := store.Get(saved.ID)
			Expect(err).NotTo(HaveOccurred())
			Expect(reloaded.Filename).To(Equal("copy.txt"))
			Expect(reloaded.Purpose).To(Equal("assistants"))
		})
	})
})
