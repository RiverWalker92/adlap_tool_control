#include "adlap_tool_control/serial_port.hpp"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

SerialPort::SerialPort(const std::string& device, int baudrate)
    : device_(device), baudrate_(baudrate), fd_(-1) {}

SerialPort::~SerialPort() {
    closePort();
}

bool SerialPort::openPort() {
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

void SerialPort::closePort() {
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

bool SerialPort::writeData(const std::string& data) {
    if (fd_ == -1) return false;
    ssize_t n = write(fd_, data.c_str(), data.size());
    return n == static_cast<ssize_t>(data.size());
}

std::string SerialPort::readData(size_t max_bytes) {
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
