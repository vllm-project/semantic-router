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
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("FileStore concurrent saves", func() {
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

	It("should handle concurrent saves safely", func() {
		var wg sync.WaitGroup
		errors := make(chan error, 20)

		for i := 0; i < 20; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				_, err := store.Save("concurrent.txt", []byte("data"), "assistants")
				if err != nil {
					errors <- err
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			Expect(err).NotTo(HaveOccurred())
		}

		records := store.List()
		Expect(records).To(HaveLen(20))
	})
})

var _ = Describe("FileStore concurrent mixed operations", func() {
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

	It("should handle concurrent reads and writes safely", func() {
		ids := make([]string, 5)
		for i := 0; i < 5; i++ {
			r, err := store.Save("file.txt", []byte("content"), "assistants")
			Expect(err).NotTo(HaveOccurred())
			ids[i] = r.ID
		}

		var wg sync.WaitGroup
		errors := make(chan error, 30)

		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				_, err := store.Read(ids[idx%5])
				if err != nil {
					errors <- err
				}
			}(i)
		}

		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, err := store.Save("new.txt", []byte("new"), "assistants")
				if err != nil {
					errors <- err
				}
			}()
		}

		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = store.List()
			}()
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			Expect(err).NotTo(HaveOccurred())
		}
	})

	It("should handle concurrent delete and get safely", func() {
		r, err := store.Save("target.txt", []byte("data"), "assistants")
		Expect(err).NotTo(HaveOccurred())

		var wg sync.WaitGroup
		wg.Add(2)

		go func() {
			defer wg.Done()
			_ = store.Delete(r.ID)
		}()
		go func() {
			defer wg.Done()
			_, _ = store.Get(r.ID)
		}()

		wg.Wait()
	})
})
