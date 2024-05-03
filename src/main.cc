#include <arm_neon.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <queue>
#include <thread>
#include <condition_variable>

class MappedFile {
  int fd = -1;
  void *addr = nullptr;
  size_t fileSize = 0;
 public:
  explicit MappedFile(const std::string &filename) {
    fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open file");
    }

    // Obtain the size of the file
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      close(fd);
      throw std::runtime_error("Failed to get file size");
    }
    fileSize = sb.st_size;

    // Map the file into memory
    addr = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to map the file");
    }
  }

  ~MappedFile() {
    if (addr) munmap(addr, fileSize);
    if (fd != -1) close(fd);
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

  MappedFile(MappedFile &&other) noexcept: fd(other.fd), addr(other.addr), fileSize(other.fileSize) {
    other.addr = nullptr;
    other.fd = -1;
    other.fileSize = 0;
  }

  MappedFile &operator=(MappedFile &&other) noexcept {
    if (this != &other) return *this;
    addr = other.addr;
    fileSize = other.fileSize;
    fd = other.fd;

    other.addr = nullptr;
    other.fileSize = 0;
    other.fd = -1;
    return *this;
  }

  [[nodiscard]] void *data() const &{ return addr; }

  [[nodiscard]] size_t size() const &{ return fileSize; }
};

static uint16_t neon_movemask(uint8x16_t input) {
  const uint8_t __attribute__((aligned(16)))
      mask_data[16] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01, 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};
  uint8x16_t mask = vld1q_u8(mask_data);
  uint8x16_t masked = vandq_u8(input, mask);
  uint8x16_t tmp = vpaddq_u8(masked, masked);
  tmp = vpaddq_u8(tmp, tmp);
  tmp = vpaddq_u8(tmp, tmp);
  return vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0);
}

std::queue<std::string> data_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
bool done = false;

void process_data() {
  std::unique_lock<std::mutex> lock(queue_mutex);
  while (!done || !data_queue.empty()) {
    if (!data_queue.empty()) {
      std::string data = data_queue.front();
      data_queue.pop();
      lock.unlock();
//      std::cout << "Processing: " << data << std::endl;
      lock.lock();
    } else {
      queue_cv.wait(lock);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Error: provide absolute path to dataset" << std::endl;
    return 1;
  }

  try {
    MappedFile file(argv[1]);
    auto data = reinterpret_cast<uint8_t *>(file.data());
    size_t start = 0;

    for (size_t i = 0; i < file.size(); i += 16) {
      uint8x16_t chunk = vld1q_u8(&data[i]);
      uint8x16_t result = vceqq_u8(chunk, vdupq_n_u8('\n'));
      uint16_t mask = neon_movemask(result);

      for (int j = 0; j < 16; ++j) {
        if (mask & (1 << j)) {
          size_t length = i + j - start;
          std::string line(reinterpret_cast<char *>(&data[start]), length);
          {
            std::lock_guard<std::mutex> lock(queue_mutex);
            data_queue.push(line);
          }
          start = i + j + 1;
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      done = true;
      std::cout << "Queue size: " << data_queue.size() << std::endl;
    }
    queue_cv.notify_all();

    std::thread worker(process_data);
    worker.join();

    std::cout << "Queue processing completed." << std::endl;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
