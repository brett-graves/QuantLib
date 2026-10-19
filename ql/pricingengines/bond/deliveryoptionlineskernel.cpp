/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026 Brett Graves

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
#include <ql/pricingengines/bond/deliveryoptionlineskernel.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace QuantLib {

    namespace {

        constexpr Real inverseSqrt2 = 0.70710678118654752440;
        constexpr Real inverseSqrt2Pi = 0.39894228040143267794;

        inline Real normalCdf(Real x) { return 0.5 * std::erfc(-x * inverseSqrt2); }

        inline Real normalPdf(Real x) { return inverseSqrt2Pi * std::exp(-0.5 * x * x); }

        // Cubic Hermite interpolant through (f0, f1) with slopes (m0, m1) per
        // unit shift, at cell position t in [0, 1].
        inline Real hermite(Real f0, Real f1, Real m0, Real m1, Real step, Real t) {
            m0 *= step;
            m1 *= step;
            const Real gap = f1 - f0;
            const Real quadratic = 3.0 * gap - 2.0 * m0 - m1;
            const Real cubic = -2.0 * gap + m0 + m1;
            return ((cubic * t + quadratic) * t + m0) * t + f0;
        }

        void requireFinite(const std::vector<Real>& v, const char* name) {
            for (Real x : v)
                QL_REQUIRE(std::isfinite(x), name << " must be finite");
        }

    }

    DeliveryOptionLinesKernel::DeliveryOptionLinesKernel(std::vector<Real> beta,
                                                         std::vector<Real> factor1,
                                                         std::vector<Real> factor2,
                                                         std::vector<Real> conversion,
                                                         std::vector<Real> adjustment,
                                                         std::vector<Real> derivativeShift,
                                                         std::vector<Real> gridLower,
                                                         Real gridStep,
                                                         const std::vector<Size>& gridLength,
                                                         std::vector<Real> prices,
                                                         std::vector<Real> slopes,
                                                         std::vector<Real> curvatures,
                                                         std::vector<Real> factor1Edges,
                                                         std::vector<Real> factor2Nodes,
                                                         std::vector<Real> factor2Weights,
                                                         Real tieTolerance)
    : beta_(std::move(beta)), factor1_(std::move(factor1)), factor2_(std::move(factor2)),
      conversion_(std::move(conversion)), adjustment_(std::move(adjustment)),
      derivativeShift_(std::move(derivativeShift)), gridLower_(std::move(gridLower)),
      gridStep_(gridStep), inverseStep_(1.0 / gridStep), gridLength_(gridLength),
      prices_(std::move(prices)), slopes_(std::move(slopes)), curvatures_(std::move(curvatures)),
      edges_(std::move(factor1Edges)), factor2Nodes_(std::move(factor2Nodes)),
      factor2Weights_(std::move(factor2Weights)), tieTolerance_(tieTolerance) {
        const Size n = beta_.size();
        QL_REQUIRE(n > 0, "at least one bond required");
        for (const auto* v : {&factor1_, &factor2_, &conversion_, &adjustment_,
                              &derivativeShift_, &gridLower_})
            QL_REQUIRE(v->size() == n, "every per-bond input needs " << n << " entries, got "
                                                                     << v->size());
        QL_REQUIRE(gridLength_.size() == n,
                   "gridLength needs " << n << " entries, got " << gridLength_.size());
        for (const auto* v : {&beta_, &factor1_, &factor2_, &conversion_, &adjustment_,
                              &derivativeShift_, &gridLower_, &edges_, &factor2Nodes_,
                              &factor2Weights_})
            requireFinite(*v, "kernel input");
        for (Real c : conversion_)
            QL_REQUIRE(c > 0.0, "conversion factors must be positive, got " << c);
        QL_REQUIRE(std::isfinite(gridStep_) && gridStep_ > 0.0,
                   "gridStep must be positive, got " << gridStep_);
        QL_REQUIRE(std::isfinite(tieTolerance_) && tieTolerance_ >= 0.0,
                   "tieTolerance must be non-negative, got " << tieTolerance_);

        gridOffset_.resize(n);
        Size total = 0;
        for (Size i = 0; i < n; ++i) {
            QL_REQUIRE(gridLength_[i] >= 2,
                       "every price table needs at least 2 points, bond " << i << " has "
                                                                          << gridLength_[i]);
            gridOffset_[i] = total;
            total += gridLength_[i];
        }
        QL_REQUIRE(prices_.size() == total && slopes_.size() == total &&
                       curvatures_.size() == total,
                   "price tables need " << total << " points each, got " << prices_.size()
                                        << ", " << slopes_.size() << ", "
                                        << curvatures_.size());
        requireFinite(prices_, "prices");
        requireFinite(slopes_, "slopes");
        requireFinite(curvatures_, "curvatures");

        QL_REQUIRE(edges_.size() >= 2, "at least one first-factor panel required");
        width_ = edges_[1] - edges_[0];
        QL_REQUIRE(width_ > 0.0, "first-factor edges must increase");
        for (Size e = 1; e < edges_.size(); ++e) {
            const Real w = edges_[e] - edges_[e - 1];
            QL_REQUIRE(std::fabs(w - width_) <= 1e-12 * std::max(1.0, std::fabs(width_)),
                       "first-factor edges must be uniform");
        }
        QL_REQUIRE(!factor2Nodes_.empty() && factor2Nodes_.size() == factor2Weights_.size(),
                   "factor2 nodes and weights must be non-empty and of equal size");

        const Size nEdges = edges_.size();
        edgeCdf_.resize(nEdges);
        edgePdf_.resize(nEdges);
        for (Size e = 0; e < nEdges; ++e) {
            edgeCdf_[e] = normalCdf(edges_[e]);
            edgePdf_[e] = normalPdf(edges_[e]);
        }
        edgeValues_.resize(nEdges * n);
        edgeDerivatives_.resize(nEdges * n);
        secondDifferences_.resize(nEdges * n);
        alpha_.resize(n);
        slope_.resize(n);
        panelMin_.resize(n);
        candidates_.resize(n);
        bound_.resize(n);
        active_.resize(n);
        // The panel correction is a mean second difference over 12; a second
        // difference of a converted price is at most its largest curvature times
        // the squared edge step in shift, and every table curvature is a value
        // the interpolant attains or bounds at the grid points.
        correctionBound_.resize(n);
        for (Size i = 0; i < n; ++i) {
            Real curvature = 0.0;
            for (Size k = gridOffset_[i]; k < gridOffset_[i] + gridLength_[i]; ++k)
                curvature = std::max(curvature, std::fabs(curvatures_[k]));
            const Real h = factor1_[i] * width_;
            correctionBound_[i] = 2.0 * curvature * h * h / (12.0 * conversion_[i]);
        }
    }

    void DeliveryOptionLinesKernel::tableAt(Size bond, Real shift, Size& index, Real& t) const {
        const Real position = (shift - gridLower_[bond]) * inverseStep_;
        const Size top = gridLength_[bond] - 2;
        // truncation toward zero, clipped into the table as the Python kernel does
        Real cell = std::trunc(position);
        if (!(cell >= 0.0))
            cell = 0.0;
        Size local = cell > static_cast<Real>(top) ? top : static_cast<Size>(cell);
        index = gridOffset_[bond] + local;
        t = position - static_cast<Real>(local);
    }

    void DeliveryOptionLinesKernel::evaluate(const std::vector<Real>& levels,
                                             bool withSensitivities) {
        const Size n = beta_.size();
        for (Real level : levels)
            QL_REQUIRE(std::isfinite(level), "levels must be finite");
        values_.assign(levels.size(), 0.0);
        ctd_.assign(levels.size() * n, 0.0);
        if (withSensitivities)
            sensitivities_.assign(levels.size() * n, 0.0);
        else
            sensitivities_.clear();
        for (Size row = 0; row < levels.size(); ++row)
            evaluateLevel(levels[row], withSensitivities, row);
    }

    Real DeliveryOptionLinesKernel::convertedPrice(Size bond, Real shift) const {
        Size k;
        Real t;
        tableAt(bond, shift, k, t);
        return (hermite(prices_[k], prices_[k + 1], slopes_[k], slopes_[k + 1], gridStep_, t) +
                adjustment_[bond]) /
               conversion_[bond];
    }

    void DeliveryOptionLinesKernel::evaluateLevel(Real level, bool withSensitivities, Size row) {
        const Size n = beta_.size();
        const Size nEdges = edges_.size();
        const Size nPanels = nEdges - 1;
        Real* ctd = ctd_.data() + row * n;
        Real* sensitivity = withSensitivities ? sensitivities_.data() + row * n : nullptr;
        Real value = 0.0;

        for (Size m = 0; m < factor2Nodes_.size(); ++m) {
            const Real weight = factor2Weights_[m];
            const Real z2 = factor2Nodes_[m];

            // Screen: a converted price is monotone in z1 (the clean price is
            // monotone in its yield shift, which is linear in z1), so its two
            // end values bound it on every panel. Lowered by the largest
            // possible curvature correction, the smaller end bounds every
            // panel minimum; the smallest larger end bounds every panel
            // ceiling. A bond whose bound clears that ceiling by more than the
            // tie tolerance is never a candidate, and dropping it changes no
            // panel's ceiling, so the screen is exact.
            Real ceiling = std::numeric_limits<Real>::infinity();
            for (Size i = 0; i < n; ++i) {
                const Real base = beta_[i] * level + factor2_[i] * z2;
                const Real first = convertedPrice(i, base + factor1_[i] * edges_.front());
                const Real last = convertedPrice(i, base + factor1_[i] * edges_.back());
                bound_[i] = std::min(first, last) - correctionBound_[i];
                ceiling = std::min(ceiling, std::max(first, last));
            }
            Size nActive = 0;
            for (Size i = 0; i < n; ++i)
                if (bound_[i] <= ceiling + tieTolerance_)
                    active_[nActive++] = i;

            // converted prices (and derivatives) of the active bonds at every edge
            for (Size a = 0; a < nActive; ++a) {
                const Size i = active_[a];
                const Real base = beta_[i] * level + factor2_[i] * z2;
                for (Size e = 0; e < nEdges; ++e) {
                    Size k;
                    Real t;
                    tableAt(i, base + factor1_[i] * edges_[e], k, t);
                    edgeValues_[e * n + i] =
                        (hermite(prices_[k], prices_[k + 1], slopes_[k], slopes_[k + 1], gridStep_,
                                 t) +
                         adjustment_[i]) /
                        conversion_[i];
                    if (withSensitivities)
                        edgeDerivatives_[e * n + i] =
                            (hermite(slopes_[k], slopes_[k + 1], curvatures_[k],
                                     curvatures_[k + 1], gridStep_, t) +
                             derivativeShift_[i]) /
                            conversion_[i];
                }
                if (nEdges >= 3) {
                    for (Size e = 1; e + 1 < nEdges; ++e)
                        secondDifferences_[e * n + i] = edgeValues_[(e + 1) * n + i] -
                                                        2.0 * edgeValues_[e * n + i] +
                                                        edgeValues_[(e - 1) * n + i];
                    secondDifferences_[i] = secondDifferences_[n + i];
                    secondDifferences_[(nEdges - 1) * n + i] =
                        secondDifferences_[(nEdges - 2) * n + i];
                } else {
                    secondDifferences_[i] = secondDifferences_[n + i] = 0.0;
                }
            }

            for (Size p = 0; p < nPanels; ++p) {
                const Real low = edges_[p];
                const Real high = edges_[p + 1];
                Real panelCeiling = std::numeric_limits<Real>::infinity();
                for (Size a = 0; a < nActive; ++a) {
                    const Size i = active_[a];
                    const Real left = edgeValues_[p * n + i];
                    const Real right = edgeValues_[(p + 1) * n + i];
                    const Real s = (right - left) / width_;
                    const Real correction =
                        (secondDifferences_[p * n + i] + secondDifferences_[(p + 1) * n + i]) /
                        24.0;
                    slope_[i] = s;
                    alpha_[i] = left - s * low - correction;
                    panelMin_[i] = std::min(left, right) - correction;
                    panelCeiling = std::min(panelCeiling, std::max(left, right));
                }
                Size count = 0;
                for (Size a = 0; a < nActive; ++a) {
                    const Size i = active_[a];
                    if (panelMin_[i] <= panelCeiling + tieTolerance_)
                        candidates_[count++] = i;
                }

                for (Size c = 0; c < count; ++c) {
                    const Size i = candidates_[c];
                    Real mass, density;
                    if (count == 1) {
                        mass = edgeCdf_[p + 1] - edgeCdf_[p];
                        density = edgePdf_[p] - edgePdf_[p + 1];
                    } else {
                        Real lo = low, hi = high;
                        bool dominated = false;
                        for (Size d = 0; d < count; ++d) {
                            const Size j = candidates_[d];
                            if (j == i)
                                continue;
                            const Real gap = slope_[j] - slope_[i];
                            if (gap > 0.0) {
                                lo = std::max(lo, (alpha_[i] - alpha_[j]) / gap);
                            } else if (gap < 0.0) {
                                hi = std::min(hi, (alpha_[i] - alpha_[j]) / gap);
                            } else if (alpha_[j] < alpha_[i] ||
                                       (alpha_[j] == alpha_[i] && j < i)) {
                                dominated = true;
                                break;
                            }
                        }
                        if (dominated || !(lo < hi))
                            continue;
                        mass = normalCdf(hi) - normalCdf(lo);
                        density = normalPdf(lo) - normalPdf(hi);
                    }
                    value += weight * (alpha_[i] * mass + slope_[i] * density);
                    ctd[i] += weight * mass;
                    if (sensitivity != nullptr) {
                        const Real left = edgeDerivatives_[p * n + i];
                        const Real s = (edgeDerivatives_[(p + 1) * n + i] - left) / width_;
                        sensitivity[i] += weight * ((left - s * low) * mass + s * density);
                    }
                }
            }
        }
        values_[row] = value;
    }

}
