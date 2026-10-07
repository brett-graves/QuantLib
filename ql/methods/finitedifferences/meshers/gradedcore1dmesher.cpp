/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026 Chloride Project

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <https://www.quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

#include <ql/errors.hpp>
#include <ql/methods/finitedifferences/meshers/gradedcore1dmesher.hpp>
#include <ql/utilities/null.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace QuantLib {

    namespace {

        /* m spacings h r, h r^2, ..., h r^m summing to length: returns
           the node offsets from the core edge (the last one == length)
           and the ratio r (1 for a uniform wing). */
        std::vector<Real> wing(Real length, Real h, Size m, Real& ratio) {
            std::vector<Real> offsets(m);
            if (m == 0)
                return offsets;
            if (h * m >= length) {
                // no growth needed: uniform at length/m <= h
                ratio = 1.0;
                for (Size i=0; i < m; ++i)
                    offsets[i] = length * Real(i + 1) / m;
                return offsets;
            }
            const auto sum = [h, m](Real r) {
                Real s = 0.0, p = 1.0;
                for (Size i=0; i < m; ++i) {
                    p *= r;
                    s += h * p;
                }
                return s;
            };
            Real lo = 1.0, hi = 2.0;
            while (sum(hi) < length)
                hi *= 2.0;
            for (Size i=0; i < 200 && hi - lo > 1e-14 * hi; ++i) {
                const Real mid = 0.5 * (lo + hi);
                (sum(mid) < length ? lo : hi) = mid;
            }
            ratio = 0.5 * (lo + hi);
            Real x = 0.0, step = h;
            for (Size i=0; i < m; ++i) {
                step *= ratio;
                x += step;
                offsets[i] = x;
            }
            offsets.back() = length;
            return offsets;
        }

    }

    GradedCore1dMesher::GradedCore1dMesher(Real start,
                                           Real end,
                                           Size size,
                                           Real center,
                                           Real coreHalfWidth,
                                           Real coreFraction)
    : Fdm1dMesher(size), ratios_(1.0, 1.0) {
        QL_REQUIRE(start < end, "start (" << start << ") must be below end ("
                                          << end << ")");
        QL_REQUIRE(start < center && center < end,
                   "center " << center << " outside (" << start << ", "
                             << end << ")");
        QL_REQUIRE(coreHalfWidth > 0.0, "core half width must be positive");
        QL_REQUIRE(coreFraction > 0.0 && coreFraction < 1.0,
                   "core fraction must be in (0, 1)");
        QL_REQUIRE(size >= 7, "at least 7 nodes required");

        // core: center + j h, j = -k..k, strictly inside (start, end)
        Size nc = std::max<Size>(3, static_cast<Size>(std::lround(coreFraction * size)));
        if (nc % 2 == 0)
            --nc;
        nc = std::min(nc, size - 4);
        if (nc % 2 == 0)
            --nc;
        const Size k = (nc - 1) / 2;
        const Real h = coreHalfWidth / k;
        std::vector<Real> core;
        for (Size i=0; i < nc; ++i) {
            const Real x = center + (Real(i) - Real(k)) * h;
            if (x > start && x < end)
                core.push_back(x);
        }

        // wings: the rest, split by length, at least one node (the end) each
        const Real lowerLength = core.front() - start;
        const Real upperLength = end - core.back();
        const Size rest = size - core.size();
        Size nl = static_cast<Size>(
            std::lround(rest * lowerLength / (lowerLength + upperLength)));
        nl = std::min(std::max<Size>(nl, 1), rest - 1);
        const Size nu = rest - nl;

        const std::vector<Real> lower = wing(lowerLength, h, nl, ratios_.first);
        const std::vector<Real> upper = wing(upperLength, h, nu, ratios_.second);

        Size i = 0;
        for (Size j = nl; j-- > 0;)
            locations_[i++] = core.front() - lower[j];
        for (Real x : core)
            locations_[i++] = x;
        for (Real d : upper)
            locations_[i++] = core.back() + d;
        locations_.front() = start;
        locations_.back() = end;

        dplus_.back() = dminus_.front() = Null<Real>();
        for (Size j=0; j < size - 1; ++j)
            dplus_[j] = dminus_[j+1] = locations_[j+1] - locations_[j];
    }

}
