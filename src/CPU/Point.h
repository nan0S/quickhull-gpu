#pragma once

#include <tuple>
#include <ostream>

struct point
{
   float x, y;
};

inline bool operator==(point u, point v)
{
   return std::tie(u.x, u.y) == std::tie(v.x, v.y);
}

inline bool operator<(point u, point v)
{
   return std::tie(u.x, u.y) < std::tie(v.x, v.y);
}

inline bool operator>(point u, point v)
{
   return std::tie(u.x, u.y) > std::tie(v.x, v.y);
}

inline point operator+(point u, point v)
{
   return { u.x + v.x, u.y + v.y };
}

inline point operator-(point u, point v)
{
   return { u.x - v.x, u.y - v.y };
}

inline point& operator+=(point& u, point v)
{
   u.x += v.x;
   u.y += v.y;
   return u;
}

inline std::ostream& operator<<(std::ostream& out, point p)
{
   return out << '(' << p.x << ',' << p.y << ')';
}

inline float cross(point u, point v)
{
   return u.x * v.y - u.y * v.x;
}

