#ifndef KEYBOARD_READER_HPP
#define KEYBOARD_READER_HPP

#include <termios.h>
#include <cstdint>

// Keycode constants
constexpr int8_t KEYCODE_RIGHT = 0x43;
constexpr int8_t KEYCODE_LEFT  = 0x44;
constexpr int8_t KEYCODE_UP    = 0x41;
constexpr int8_t KEYCODE_DOWN  = 0x42;
constexpr int8_t KEYCODE_SPACE = 0x20;
constexpr int8_t KEYCODE_ENTER = 0x0A;
constexpr int8_t KEYCODE_A = 0x61;
constexpr int8_t KEYCODE_B = 0x62;
constexpr int8_t KEYCODE_C = 0x63;
constexpr int8_t KEYCODE_D = 0x64;
constexpr int8_t KEYCODE_E = 0x65;
constexpr int8_t KEYCODE_F = 0x66;
constexpr int8_t KEYCODE_G = 0x67;
constexpr int8_t KEYCODE_H = 0x68;
constexpr int8_t KEYCODE_I = 0x69;
constexpr int8_t KEYCODE_J = 0x6A;
constexpr int8_t KEYCODE_K = 0x6B;
constexpr int8_t KEYCODE_L = 0x6C;
constexpr int8_t KEYCODE_M = 0x6D;
constexpr int8_t KEYCODE_N = 0x6E;
constexpr int8_t KEYCODE_O = 0x6F;
constexpr int8_t KEYCODE_P = 0x70;  
constexpr int8_t KEYCODE_Q = 0x71;
constexpr int8_t KEYCODE_R = 0x72;
constexpr int8_t KEYCODE_S = 0x73;
constexpr int8_t KEYCODE_T = 0x74;
constexpr int8_t KEYCODE_U = 0x75;
constexpr int8_t KEYCODE_V = 0x76;
constexpr int8_t KEYCODE_W = 0x77;
constexpr int8_t KEYCODE_X = 0x78;
constexpr int8_t KEYCODE_Y = 0x79;
constexpr int8_t KEYCODE_Z = 0x7A;

// A class for reading key inputs from the terminal
class KeyboardReader
{
public:
  KeyboardReader();
  void readOne(char* c);
  void shutdown();

private:
  int file_descriptor_;
  struct termios cooked_;
};

#endif // KEYBOARD_READER_HPP
