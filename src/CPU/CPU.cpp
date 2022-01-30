#include "CPU.h"

#include <vector>
#include <random>
#include <tuple>
#include <algorithm>

#include "Utils/Timer.h"
#include "Utils/Log.h"
#include "Graphics/Error.h"

namespace CPU
{
    /* macros */
	#define PI 3.14159265358f

    /* structs */
	struct point
	{
		float x, y;

		static float cross(point u, point v);

        bool operator<(point p) const;
        bool operator>(point p) const;
        bool operator==(point p) const;
		point operator-(point p) const;
		point operator+(point p) const;
		point& operator+=(point p);

		friend std::ostream& operator<<(std::ostream& out, const point& p);
	};

    struct CPUGenerator 
    {
        std::mt19937 rng;
        std::uniform_real_distribution<float> adist;
        std::uniform_real_distribution<float> rdist;
    };

    struct Memory
    {
        point* points;
        point* buffer;
    };

    /* forward declarations */
    int quickHull(point* first, point* last);
	point* findHull(point* first, point* last, point u, point v);
	template<class Pred>
	point* half_stable_partition(point* first, point* last, Pred pred);
	int grahamScan(point* ps, int n);

    /* variables */
    CPUGenerator cpu_gen;
    Memory mem;

	void init(Config config, const std::vector<int>& num_points)
	{
		float r_min = 0.f, r_max = 1.f;
		switch (config.dataset_type)
		{
			case DatasetType::DISC:
				r_min = 0.f; r_max = 1.f;
				break;
			case DatasetType::RING:
				r_min = 0.9f; r_max = 1.f;
				break;
			case DatasetType::CIRCLE:
				r_min = 1.f; r_max = 1.f;
				break;
			default:
				assert(false);
		}
        cpu_gen.rng.seed(config.seed);
        cpu_gen.adist.param(decltype(cpu_gen.adist)::param_type(0, 2 * PI));
        cpu_gen.rdist.param(decltype(cpu_gen.rdist)::param_type(r_min, r_max));

		int max_n = *std::max_element(num_points.begin(), num_points.end());
		mem.points = new point[max_n];
		mem.buffer = new point[max_n];

		size_t bytes = max_n * sizeof(point);
		glCall(glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_STATIC_DRAW));
		glCall(glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, sizeof(point),
			(const void*)offsetof(point, x)));
		glCall(glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(point),
			(const void*)offsetof(point, y)));
	}

	int calculate(int n)
	{
		print("\nRunning CPU for ", n, " points.");

		// Generate points.
		for (int i = 0; i < n; ++i)
		{
			float a = cpu_gen.adist(cpu_gen.rng), r = cpu_gen.rdist(cpu_gen.rng);
			float x = r * cos(a);
			float y = r * sin(a);
			mem.points[i] = mem.buffer[i] = { x, y };
		}

		int hull_count;
		{
			Timer timer("Graham Scan");
			grahamScan(mem.points, n);
		}
        std::copy(mem.buffer, mem.buffer + n, mem.points);
		{
			Timer timer("QuickHull");
			hull_count = quickHull(mem.points, mem.points + n);
		}

		glCall(glBufferSubData(GL_ARRAY_BUFFER, 0, n * sizeof(point), mem.points));

		return hull_count;
	}

	void cleanup()
	{
		delete[] mem.points;
		delete[] mem.buffer;
	}

	float point::cross(point u, point v)
	{
		return u.x * v.y - u.y * v.x;
	}

    bool point::operator<(point p) const
    {
        return std::tie(x, y) < std::tie(p.x, p.y);
    }

    bool point::operator>(point p) const
    {
        return std::tie(x, y) > std::tie(p.x, p.y);
    }

    bool point::operator==(point p) const
    {
        return std::tie(x, y) == std::tie(p.x, p.y);
    }

	point point::operator-(point p) const
	{
		return { x - p.x, y - p.y };
	}

	point point::operator+(point p) const
	{
		return { x + p.x, y + p.y };
	}

	point& point::operator+=(point p)
	{
		x += p.x; y += p.y;
		return *this;
	}

	std::ostream& operator<<(std::ostream& out, const point& p)
	{
		return out << '(' << p.x << ',' << p.y << ')';
	}

	point* findHull(point* first, point* last, point u, point v)
	{
		assert(*first == u);
		if (first + 1 == last)
			return last;

		point d = v - u, * far = nullptr;
		float dist = -1;
		// We can skip the first one as it is partition boundary (u).
		// Also at this point, we are sure there are at least 2 points.
		for (point* it = first + 1; it != last; ++it)
		{
			float cur_dist = point::cross(d, u - *it);
			assert(cur_dist >= 0);
			if (cur_dist > dist)
			{
				dist = cur_dist;
				far = it;
			}
		}

		point far_p = *far, uf = u - far_p, vf = v - far_p;
		auto isOuterior = [uf, vf, far_p](point p) {
			point pf = p - far_p;
			return point::cross(pf, vf) > 0 || point::cross(uf, pf) > 0;
		};
		point* pivot = half_stable_partition(first + 1, far, isOuterior);
		point* left_boundary = findHull(first, pivot, u, far_p);
		pivot = half_stable_partition(far + 1, last, isOuterior);
		point* right_boundary = findHull(far, pivot, far_p, v);
		point* boundary = std::swap_ranges(far, right_boundary, left_boundary);

		return boundary;
	}

	int quickHull(point* first, point* last)
	{
		auto [min, max] = std::minmax_element(first, last);
		point left = *min, right = *max;
		point v = left - right;
		point* pivot = std::partition(first, last, [right, v](point p) {
			if (p == right)
				return true;
			return point::cross(p - right, v) > 0;
		});
        
		std::sort(first, pivot, std::greater<>());
		std::sort(pivot, last);
		assert(*first == right);
		assert(*pivot == left);

		point* left_boundary = findHull(first, pivot, right, left);
		point* right_boundary = findHull(pivot, last, left, right);
		// Edge case: it might happen that point right after pivot
		// is on the line between left and right. In that case remove it.
		if (pivot + 1 != last)
		{
			point p = pivot[1];
			if (point::cross(right - left, p - left) == 0)
			{
				assert(pivot + 2 == right_boundary);
				--right_boundary;
			}
		}
		point* boundary = std::swap_ranges(pivot, right_boundary, left_boundary);

		return static_cast<int>(boundary - first);
	}

	template<class Pred>
	point* half_stable_partition(point* first, point* last, Pred pred)
	{
		point* pivot = first;
		for (point* it = first; it != last; ++it)
			if (pred(*it))
				std::swap(*it, *pivot++);
		return pivot;
	}

	int grahamScan(point* ps, int n)
	{
		point* min_it = std::min_element(ps, ps + n);
		point min = *min_it;
		std::swap(*min_it, ps[0]);
		std::transform(ps, ps + n, ps, [min](point p) {
			return p - min;
		});
		std::sort(ps + 1, ps + n, [](point u, point v) {
			float c = point::cross(u, v);
			if (c != 0)
				return c > 0;
			if (u.x != v.x)
				return u.x < v.x;
			return u.y < v.y;
		});

		int m = 1;
		for (int i = 1; i < n; ++i)
		{
			point u = ps[i], v;
			while (i < n - 1)
			{
				v = ps[i + 1];
				if (point::cross(u, v) != 0)
					break;
				u = v;
				++i;
			}
			ps[m++] = u;
		}

		point p = ps[0];
		point v = ps[1] - p;
		int s = 1;
		for (int i = 2; i < m; ++i)
		{
			point cur = ps[i];
			while (s > 0 && point::cross(cur - p, v) >= 0)
			{
				--s;
				p = ps[s - 1];
				v = ps[s] - p;
			}
			std::swap(ps[i], ps[++s]);
			p += v;
			v = cur - p;
		}
		std::transform(ps, ps + n, ps, [min](point p) {
			return p + min;
		});

		return s + 1;
	}

} // namespace CPU
