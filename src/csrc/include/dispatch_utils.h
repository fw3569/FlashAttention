#pragma once

#define BOOL_DISPATCHER(template_name, condition, ...) \
  {                                                    \
    if (condition) {                                   \
      constexpr bool template_name = true;             \
      __VA_ARGS__();                                   \
    } else {                                           \
      constexpr bool template_name = false;            \
      __VA_ARGS__();                                   \
    }                                                  \
  }

#define INT_DISPATCHER(template_name, value, candidate, ...) \
  {                                                          \
    if (value <= candidate) {                                \
      constexpr int template_name = candidate;               \
      __VA_ARGS__();                                         \
    }                                                        \
  }

#define INT_DISPATCHER_2(template_name, value, candidate0, candidate1, ...) \
  {                                                                         \
    if (value <= candidate0) {                                              \
      constexpr int template_name = candidate0;                             \
      __VA_ARGS__();                                                        \
    } else {                                                                \
      INT_DISPATCHER(template_name, value, candidate1, __VA_ARGS__)         \
    }                                                                       \
  }

#define INT_DISPATCHER_3(template_name, value, candidate0, candidate1, \
                         candidate2, ...)                              \
  {                                                                    \
    if (value <= candidate0) {                                         \
      constexpr int template_name = candidate0;                        \
      __VA_ARGS__();                                                   \
    } else {                                                           \
      INT_DISPATCHER_2(template_name, value, candidate1, candidate2,   \
                       __VA_ARGS__)                                    \
    }                                                                  \
  }

#define INT_DISPATCHER_4(template_name, value, candidate0, candidate1, \
                         candidate2, candidate3, ...)                  \
  {                                                                    \
    if (value <= candidate0) {                                         \
      constexpr int template_name = candidate0;                        \
      __VA_ARGS__();                                                   \
    } else {                                                           \
      INT_DISPATCHER_3(template_name, value, candidate1, candidate2,   \
                       candidate3, __VA_ARGS__)                        \
    }                                                                  \
  }

#define INT_DISPATCHER_5(template_name, value, candidate0, candidate1, \
                         candidate2, candidate3, candidate4, ...)      \
  {                                                                    \
    if (value <= candidate0) {                                         \
      constexpr int template_name = candidate0;                        \
      __VA_ARGS__();                                                   \
    } else {                                                           \
      INT_DISPATCHER_4(template_name, value, candidate1, candidate2,   \
                       candidate3, candidate4, __VA_ARGS__)            \
    }                                                                  \
  }

#define INT_DISPATCHER_6(template_name, value, candidate0, candidate1,        \
                         candidate2, candidate3, candidate4, candidate5, ...) \
  {                                                                           \
    if (value <= candidate0) {                                                \
      constexpr int template_name = candidate0;                               \
      __VA_ARGS__();                                                          \
    } else {                                                                  \
      INT_DISPATCHER_5(template_name, value, candidate1, candidate2,          \
                       candidate3, candidate4, candidate5, __VA_ARGS__)       \
    }                                                                         \
  }

#define INT_DISPATCHER_7(template_name, value, candidate0, candidate1,   \
                         candidate2, candidate3, candidate4, candidate5, \
                         candidate6, ...)                                \
  {                                                                      \
    if (value <= candidate0) {                                           \
      constexpr int template_name = candidate0;                          \
      __VA_ARGS__();                                                     \
    } else {                                                             \
      INT_DISPATCHER_6(template_name, value, candidate1, candidate2,     \
                       candidate3, candidate4, candidate5, candidate6,   \
                       __VA_ARGS__)                                      \
    }                                                                    \
  }

#define INT_DISPATCHER_8(template_name, value, candidate0, candidate1,   \
                         candidate2, candidate3, candidate4, candidate5, \
                         candidate6, candidate7, ...)                    \
  {                                                                      \
    if (value <= candidate0) {                                           \
      constexpr int template_name = candidate0;                          \
      __VA_ARGS__();                                                     \
    } else {                                                             \
      INT_DISPATCHER_7(template_name, value, candidate1, candidate2,     \
                       candidate3, candidate4, candidate5, candidate6,   \
                       candidate7, __VA_ARGS__)                          \
    }                                                                    \
  }
