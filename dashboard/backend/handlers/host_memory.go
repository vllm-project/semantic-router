package handlers

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// hostMemoryBytes reports the memory the dashboard's container runtime can
// use: the smaller of the cgroup limit and the kernel's MemTotal. Under Docker
// Desktop MemTotal is the Docker VM's memory. It returns 0 when neither is
// readable, such as outside Linux.
func hostMemoryBytes() int64 {
	return hostMemoryBytesAt("/")
}

func hostMemoryBytesAt(root string) int64 {
	var smallest int64
	for _, value := range []int64{
		readCgroupLimit(filepath.Join(root, "sys/fs/cgroup/memory.max")),
		readCgroupLimit(filepath.Join(root, "sys/fs/cgroup/memory/memory.limit_in_bytes")),
		readMemTotal(filepath.Join(root, "proc/meminfo")),
	} {
		if value > 0 && (smallest == 0 || value < smallest) {
			smallest = value
		}
	}
	return smallest
}

// readCgroupLimit returns 0 for a missing file or an unlimited cgroup ("max").
// A cgroup v1 "unlimited" value is a huge number that MemTotal undercuts.
func readCgroupLimit(path string) int64 {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	value, err := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	if err != nil || value <= 0 {
		return 0
	}
	return value
}

func readMemTotal(path string) int64 {
	file, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 || fields[0] != "MemTotal:" {
			continue
		}
		kib, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil || kib <= 0 {
			return 0
		}
		return kib * 1024
	}
	return 0
}
