#include "adlap_tool_control/serial_port.hpp"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <filesystem>
#include <fstream>

SerialPort::SerialPort(const std::string& device, int baudrate)
    : device_(device), baudrate_(baudrate), fd_(-1) {}

SerialPort::~SerialPort() {
    close_port();
}

bool SerialPort::open_port() {
    fd_ = open(device_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::cerr << "Error opening serial port " << device_ << "\n";
        return false;
    }

    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
        std::cerr << "Error getting terminal attributes\n";
        return false;
    }

    speed_t speed;
    switch (baudrate_) {
        case 9600: speed = B9600; break;
        case 19200: speed = B19200; break;
        case 38400: speed = B38400; break;
        case 57600: speed = B57600; break;
        case 115200: speed = B115200; break;
        default:
            std::cerr << "Unsupported baudrate\n";
            return false;
    }

    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_iflag = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1; // 0.1 seconds timeout

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        std::cerr << "Error setting terminal attributes\n";
        return false;
    }

    return true;
}

void SerialPort::close_port() {
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

bool SerialPort::write_data(const std::string& data) {
    if (fd_ == -1) return false;
    ssize_t n = write(fd_, data.c_str(), data.size());
    return n == static_cast<ssize_t>(data.size());
}

std::string SerialPort::read_data(size_t max_bytes) {
    if (fd_ == -1) return "";

    char buffer[1024];
    size_t bytes_to_read = std::min(max_bytes, sizeof(buffer));
    ssize_t n = read(fd_, buffer, bytes_to_read);
    if (n > 0) {
        return std::string(buffer, n);
    } else {
        return "";
    }
}

std::string SerialPort::find_device_by_manufacturer_product(const std::string& manufacturer, const std::string& product) {
    const std::string sys_class_tty = "/sys/class/tty/";
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(sys_class_tty)) {
            if (!entry.is_directory()) {
                continue;
            }
            
            std::string device_name = entry.path().filename();
            
            // Only check ttyACM* and ttyUSB* devices
            if (device_name.substr(0, 6) != "ttyACM" && device_name.substr(0, 6) != "ttyUSB") {
                continue;
            }
            
            // Check if device has USB info
            std::filesystem::path device_path = entry.path() / "device";
            if (!std::filesystem::exists(device_path)) {
                continue;
            }
            
            // Follow symlinks to find the USB device info
            std::filesystem::path usb_device_path = device_path;
            try {
                while (std::filesystem::is_symlink(usb_device_path)) {
                    std::filesystem::path link_target = std::filesystem::read_symlink(usb_device_path);
                    if (link_target.is_relative()) {
                        usb_device_path = usb_device_path.parent_path() / link_target;
                        usb_device_path = std::filesystem::canonical(usb_device_path);
                    } else {
                        usb_device_path = link_target;
                    }
                }
            } catch (const std::filesystem::filesystem_error&) {
                continue; // Skip if we can't resolve the symlink
            }
            
            // Navigate up to find USB device directory with manufacturer and product files
            std::filesystem::path current_path = usb_device_path;
            while (current_path != current_path.parent_path()) {
                std::filesystem::path manufacturer_file = current_path / "manufacturer";
                std::filesystem::path product_file = current_path / "product";
                
                if (std::filesystem::exists(manufacturer_file) && std::filesystem::exists(product_file)) {
                    // Read manufacturer
                    std::ifstream mfg_stream(manufacturer_file);
                    std::string device_manufacturer;
                    if (mfg_stream.is_open()) {
                        std::getline(mfg_stream, device_manufacturer);
                    }
                    
                    // Read product
                    std::ifstream prod_stream(product_file);
                    std::string device_product;
                    if (prod_stream.is_open()) {
                        std::getline(prod_stream, device_product);
                    }
                    
                    // Check if this matches our target device
                    if (device_manufacturer.find(manufacturer) != std::string::npos && 
                        device_product.find(product) != std::string::npos) {
                        return "/dev/" + device_name;
                    }
                    break;
                }
                current_path = current_path.parent_path();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error searching for device: " << e.what() << std::endl;
    }
    
    // Return empty string if not found
    return "";
}
