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

#include <ql/termstructures/volatility/equityfx/parametricvoltermstructure.hpp>
#include <ql/cashflows/dividend.hpp>
#include <ql/errors.hpp>
#include <ql/utilities/null.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace QuantLib {

    // ===== ParametricVolShape default d2fdz2 / dfdParams (central FD) ========

    Real ParametricVolShape::d2fdz2(Real z,
                                    const std::vector<Real>& params,
                                    Real h) const {
        // Central FD on dfdz.  Goes through the vtable so Python
        // proxies (which override dfdz) work transparently.
        return (dfdz(z + h, params) - dfdz(z - h, params)) / (2.0 * h);
    }

    std::vector<Real> ParametricVolShape::dfdParams(
            Real z, const std::vector<Real>& params) const {
        std::vector<Real> out(params.size());
        std::vector<Real> p = params;
        for (std::size_t i = 0; i < params.size(); ++i) {
            Real h = std::max(std::abs(params[i]) * 1e-5, 1e-7);
            p[i] = params[i] + h;
            Real f_plus = f(z, p);
            p[i] = params[i] - h;
            Real f_minus = f(z, p);
            p[i] = params[i];
            out[i] = (f_plus - f_minus) / (2.0 * h);
        }
        return out;
    }

    // ===== S3Shape ============================================================

    Real S3Shape::f(Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 2,
                   "S3Shape: expected 2 params (s2, c2), got " << p.size());
        Real s2 = p[0], c2 = p[1];
        QL_REQUIRE(c2 >= 0.0,
                   "S3Shape: c2 (=" << c2 << ") must be >= 0 for positivity");
        Real a = 1.0 + s2 * z;
        Real radicand = 0.25 * a * a + 0.5 * c2 * z * z;
        return 0.5 * a + std::sqrt(radicand);
    }

    Real S3Shape::dfdz(Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 2,
                   "S3Shape: expected 2 params (s2, c2), got " << p.size());
        Real s2 = p[0], c2 = p[1];
        Real a = 1.0 + s2 * z;
        Real radicand = 0.25 * a * a + 0.5 * c2 * z * z;
        Real root = std::sqrt(radicand);
        // d/dz [0.5 a]       = 0.5 s2
        // d/dz [sqrt(rad)]   = (0.5 a s2 + c2 z) / (2 root)
        return 0.5 * s2 + (0.5 * a * s2 + c2 * z) / (2.0 * root);
    }

    Real S3Shape::d2fdz2(Real z, const std::vector<Real>& p, Real /*h*/) const {
        QL_REQUIRE(p.size() == 2,
                   "S3Shape: expected 2 params (s2, c2), got " << p.size());
        Real s2 = p[0], c2 = p[1];
        Real a = 1.0 + s2 * z;
        Real R  = 0.25 * a * a + 0.5 * c2 * z * z;
        Real r  = std::sqrt(R);
        Real Rp  = 0.5 * a * s2 + c2 * z;       // R'
        Real Rpp = 0.5 * s2 * s2 + c2;          // R''
        // 0.5·a is linear in z so contributes nothing to f''.
        // f''(z) = R''/(2 r) − R'² / (4 r³)
        return Rpp / (2.0 * r) - (Rp * Rp) / (4.0 * r * R);
    }

    std::vector<Real> S3Shape::dfdParams(
            Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 2,
                   "S3Shape: expected 2 params (s2, c2), got " << p.size());
        Real s2 = p[0], c2 = p[1];
        Real a = 1.0 + s2 * z;
        Real R = 0.25 * a * a + 0.5 * c2 * z * z;
        Real sqrtR = std::sqrt(R);
        // ∂f/∂s2 = 0.5·z + (0.5·a·z) / (2·sqrtR) = 0.5·z + 0.25·a·z/sqrtR
        // ∂f/∂c2 = (0.5·z²) / (2·sqrtR) = z²/(4·sqrtR)
        Real ds2 = 0.5 * z + 0.25 * a * z / sqrtR;
        Real dc2 = z * z / (4.0 * sqrtR);
        return {ds2, dc2};
    }


    // ===== JWShape ============================================================

    namespace {
        // S3-with-given-c kernel, used by both halves of JW.
        inline Real s3KernelF(Real z, Real s2, Real c) {
            Real a = 1.0 + s2 * z;
            Real radicand = 0.25 * a * a + 0.5 * c * z * z;
            return 0.5 * a + std::sqrt(radicand);
        }
        inline Real s3KernelDfdz(Real z, Real s2, Real c) {
            Real a = 1.0 + s2 * z;
            Real radicand = 0.25 * a * a + 0.5 * c * z * z;
            Real root = std::sqrt(radicand);
            return 0.5 * s2 + (0.5 * a * s2 + c * z) / (2.0 * root);
        }
        inline Real s3KernelD2fdz2(Real z, Real s2, Real c) {
            Real a = 1.0 + s2 * z;
            Real R  = 0.25 * a * a + 0.5 * c * z * z;
            Real r  = std::sqrt(R);
            Real Rp  = 0.5 * a * s2 + c * z;
            Real Rpp = 0.5 * s2 * s2 + c;
            return Rpp / (2.0 * r) - (Rp * Rp) / (4.0 * r * R);
        }
    }

    Real JWShape::f(Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 3,
                   "JWShape: expected 3 params (s2, c_minus, c_plus), got "
                   << p.size());
        Real s2 = p[0], cm = p[1], cp = p[2];
        QL_REQUIRE(cm >= 0.0 && cp >= 0.0,
                   "JWShape: c_minus (" << cm << ") and c_plus (" << cp
                   << ") must be >= 0");
        return s3KernelF(z, s2, (z < 0.0 ? cm : cp));
    }

    Real JWShape::dfdz(Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 3,
                   "JWShape: expected 3 params (s2, c_minus, c_plus), got "
                   << p.size());
        Real s2 = p[0], cm = p[1], cp = p[2];
        return s3KernelDfdz(z, s2, (z < 0.0 ? cm : cp));
    }

    Real JWShape::d2fdz2(Real z, const std::vector<Real>& p, Real /*h*/) const {
        QL_REQUIRE(p.size() == 3,
                   "JWShape: expected 3 params (s2, c_minus, c_plus), got "
                   << p.size());
        Real s2 = p[0], cm = p[1], cp = p[2];
        // Piecewise-S3.  f''(0) is discontinuous when cm != cp; right-
        // continuous convention matches f / dfdz at z = 0 (uses c_plus).
        return s3KernelD2fdz2(z, s2, (z < 0.0 ? cm : cp));
    }

    std::vector<Real> JWShape::dfdParams(
            Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 3,
                   "JWShape: expected 3 params (s2, c_minus, c_plus), got "
                   << p.size());
        Real s2 = p[0], cm = p[1], cp = p[2];
        Real c_eff = (z < 0.0) ? cm : cp;
        Real a = 1.0 + s2 * z;
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        Real sqrtR = std::sqrt(R);
        Real ds2 = 0.5 * z + 0.25 * a * z / sqrtR;
        Real dc = z * z / (4.0 * sqrtR);
        if (z < 0.0)
            return {ds2, dc, 0.0};
        else
            return {ds2, 0.0, dc};
    }


    // ===== K5Shape ============================================================

    namespace {
        constexpr Real K5_MU = 2.0;   // fixed tanh^2 blend width

        struct K5Blend {
            Real T;       // tanh^2(z/mu)
            Real Tp;      // d/dz tanh^2(z/mu) = (2/mu) tanh(u) sech^2(u)
            Real Tpp;     // d^2/dz^2 tanh^2(z/mu) = (2/mu^2) sech^2 (1 - 3 tanh^2)
        };

        inline K5Blend k5BlendAt(Real z) {
            Real u = z / K5_MU;
            Real th = std::tanh(u);
            Real sech2 = 1.0 - th * th;
            K5Blend b;
            b.T   = th * th;
            b.Tp  = (2.0 / K5_MU) * th * sech2;
            b.Tpp = (2.0 / (K5_MU * K5_MU)) * sech2 * (1.0 - 3.0 * b.T);
            return b;
        }

        inline void k5CEff(Real z, Real c, Real cm, Real cp,
                           Real& c_eff, Real& dc_dz, Real& d2c_dz2) {
            K5Blend b = k5BlendAt(z);
            Real c_wing = (z < 0.0) ? cm : cp;
            c_eff   = c + (c_wing - c) * b.T;
            dc_dz   = (c_wing - c) * b.Tp;
            d2c_dz2 = (c_wing - c) * b.Tpp;
        }
    }

    Real K5Shape::f(Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 4,
                   "K5Shape: expected 4 params (s, c, c_minus, c_plus), got "
                   << p.size());
        Real s = p[0];
        // Clamp curvature params to >= 0 to tolerate finite-difference
        // optimizer probes that occasionally cross zero (chloride
        // log-transforms but the FD step lands the raw param slightly
        // negative).
        Real c  = std::max(p[1], 0.0);
        Real cm = std::max(p[2], 0.0);
        Real cp = std::max(p[3], 0.0);
        Real a = 1.0 + s * z;
        Real c_eff, dc_dz, d2c_dz2;
        k5CEff(z, c, cm, cp, c_eff, dc_dz, d2c_dz2);
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        return 0.5 * a + std::sqrt(std::max(R, 0.0));
    }

    Real K5Shape::dfdz(Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 4,
                   "K5Shape: expected 4 params (s, c, c_minus, c_plus), got "
                   << p.size());
        Real s = p[0];
        Real c  = std::max(p[1], 0.0);
        Real cm = std::max(p[2], 0.0);
        Real cp = std::max(p[3], 0.0);
        Real a = 1.0 + s * z;
        Real c_eff, dc_dz, d2c_dz2;
        k5CEff(z, c, cm, cp, c_eff, dc_dz, d2c_dz2);
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        if (R <= 0.0)
            return 0.5 * s;
        Real Rp = 0.5 * s * a + 0.5 * dc_dz * z * z + c_eff * z;
        return 0.5 * s + Rp / (2.0 * std::sqrt(R));
    }

    Real K5Shape::d2fdz2(Real z, const std::vector<Real>& p,
                         Real /*h*/) const {
        QL_REQUIRE(p.size() == 4,
                   "K5Shape: expected 4 params (s, c, c_minus, c_plus), got "
                   << p.size());
        Real s = p[0];
        Real c  = std::max(p[1], 0.0);
        Real cm = std::max(p[2], 0.0);
        Real cp = std::max(p[3], 0.0);
        Real a = 1.0 + s * z;
        Real c_eff, dc_dz, d2c_dz2;
        k5CEff(z, c, cm, cp, c_eff, dc_dz, d2c_dz2);
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        if (R <= 0.0)
            return 0.0;
        Real r = std::sqrt(R);
        Real Rp = 0.5 * s * a + 0.5 * dc_dz * z * z + c_eff * z;
        Real Rpp = 0.5 * s * s + 0.5 * d2c_dz2 * z * z
                   + 2.0 * dc_dz * z + c_eff;
        return Rpp / (2.0 * r) - (Rp * Rp) / (4.0 * r * R);
    }

    std::vector<Real> K5Shape::dfdParams(
            Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 4,
                   "K5Shape: expected 4 params (s, c, c_minus, c_plus), got "
                   << p.size());
        Real s = p[0];
        Real c  = std::max(p[1], 0.0);
        Real cm = std::max(p[2], 0.0);
        Real cp = std::max(p[3], 0.0);
        Real a = 1.0 + s * z;
        Real c_eff, dc_dz, d2c_dz2;
        k5CEff(z, c, cm, cp, c_eff, dc_dz, d2c_dz2);
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        Real sqrtR = (R > 0.0) ? std::sqrt(R) : 1.0;
        K5Blend b = k5BlendAt(z);

        // f = (1/2) a + sqrt(R).
        // d/ds : 0.5 a contributes 0.5 z; R contributes 0.5 a z / (2 sqrtR).
        Real ds = 0.5 * z + 0.25 * a * z / sqrtR;
        // c_eff = (1-T) c + T c_wing, so:
        //   d c_eff / d c       = 1 - T
        //   d c_eff / d c_minus = T  (only when z<0; else 0)
        //   d c_eff / d c_plus  = T  (only when z>=0; else 0)
        // d R / d q = (1/2) z^2 (d c_eff / d q)
        // d f / d q = d R / d q  /  (2 sqrtR)
        Real dc      = z * z * (1.0 - b.T) / (4.0 * sqrtR);
        Real dcm_pos = z * z * b.T         / (4.0 * sqrtR);
        Real dcm = (z <  0.0) ? dcm_pos : 0.0;
        Real dcp = (z >= 0.0) ? dcm_pos : 0.0;
        return {ds, dc, dcm, dcp};
    }


    // ===== K7Shape ============================================================

    namespace {
        // Fixed blend widths.  Could be made per-slice params in a future K9.
        constexpr Real K7_MU_1 = 0.5;   // inner-wing knot
        constexpr Real K7_MU_2 = 2.0;   // outer-wing knot

        struct K7Blend {
            Real T1, T2;             // tanh^2(z/mu_k)
            Real T1p, T2p;           // d/dz tanh^2(z/mu_k)
            Real T1pp, T2pp;         // d^2/dz^2 tanh^2(z/mu_k)
        };

        inline K7Blend k7BlendAt(Real z) {
            Real u1 = z / K7_MU_1;
            Real u2 = z / K7_MU_2;
            Real th1 = std::tanh(u1);
            Real th2 = std::tanh(u2);
            Real sech2_1 = 1.0 - th1 * th1;
            Real sech2_2 = 1.0 - th2 * th2;
            K7Blend b;
            b.T1 = th1 * th1;
            b.T2 = th2 * th2;
            // d/dz tanh^2(z/mu) = (2/mu) * tanh(z/mu) * sech^2(z/mu)
            b.T1p = (2.0 / K7_MU_1) * th1 * sech2_1;
            b.T2p = (2.0 / K7_MU_2) * th2 * sech2_2;
            // d^2/dz^2 tanh^2(z/mu) = (2/mu^2) * sech^2(z/mu) * (1 - 3*tanh^2)
            b.T1pp = (2.0 / (K7_MU_1 * K7_MU_1)) * sech2_1
                     * (1.0 - 3.0 * b.T1);
            b.T2pp = (2.0 / (K7_MU_2 * K7_MU_2)) * sech2_2
                     * (1.0 - 3.0 * b.T2);
            return b;
        }

        inline void k7CEff(Real z,
                           Real c, Real cmm, Real cmp, Real cfm, Real cfp,
                           Real& c_eff, Real& dc_dz, Real& d2c_dz2) {
            K7Blend b = k7BlendAt(z);
            Real c_mid = (z < 0.0) ? cmm : cmp;
            Real c_far = (z < 0.0) ? cfm : cfp;
            c_eff   = c + (c_mid - c) * b.T1 + (c_far - c_mid) * b.T2;
            dc_dz   = (c_mid - c) * b.T1p + (c_far - c_mid) * b.T2p;
            d2c_dz2 = (c_mid - c) * b.T1pp + (c_far - c_mid) * b.T2pp;
        }
    }

    Real K7Shape::f(Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 6,
                   "K7Shape: expected 6 params "
                   "(s, c, c_mid_minus, c_mid_plus, c_far_minus, c_far_plus), got "
                   << p.size());
        Real s = p[0];
        // Clamp curvature params to >= 0 to tolerate optimizer rounding
        // (log-transformed in chloride's spec; tiny negatives can leak
        // through during finite-difference jacobian probes).
        Real c   = std::max(p[1], 0.0);
        Real cmm = std::max(p[2], 0.0);
        Real cmp = std::max(p[3], 0.0);
        Real cfm = std::max(p[4], 0.0);
        Real cfp = std::max(p[5], 0.0);
        Real a = 1.0 + s * z;
        Real c_eff, dc_dz, d2c_dz2;
        k7CEff(z, c, cmm, cmp, cfm, cfp, c_eff, dc_dz, d2c_dz2);
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        return 0.5 * a + std::sqrt(std::max(R, 0.0));
    }

    Real K7Shape::dfdz(Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 6, "K7Shape: expected 6 params");
        Real s = p[0];
        Real c   = std::max(p[1], 0.0);
        Real cmm = std::max(p[2], 0.0);
        Real cmp = std::max(p[3], 0.0);
        Real cfm = std::max(p[4], 0.0);
        Real cfp = std::max(p[5], 0.0);
        Real a = 1.0 + s * z;
        Real c_eff, dc_dz, d2c_dz2;
        k7CEff(z, c, cmm, cmp, cfm, cfp, c_eff, dc_dz, d2c_dz2);
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        if (R <= 0.0)
            return 0.5 * s;
        // R' = (1/2) s (1 + s z) + (1/2) c_eff' z^2 + c_eff z
        Real Rp = 0.5 * s * a + 0.5 * dc_dz * z * z + c_eff * z;
        return 0.5 * s + Rp / (2.0 * std::sqrt(R));
    }

    Real K7Shape::d2fdz2(Real z, const std::vector<Real>& p,
                         Real /*h*/) const {
        QL_REQUIRE(p.size() == 6, "K7Shape: expected 6 params");
        Real s = p[0];
        Real c   = std::max(p[1], 0.0);
        Real cmm = std::max(p[2], 0.0);
        Real cmp = std::max(p[3], 0.0);
        Real cfm = std::max(p[4], 0.0);
        Real cfp = std::max(p[5], 0.0);
        Real a = 1.0 + s * z;
        Real c_eff, dc_dz, d2c_dz2;
        k7CEff(z, c, cmm, cmp, cfm, cfp, c_eff, dc_dz, d2c_dz2);
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        if (R <= 0.0)
            return 0.0;
        Real r = std::sqrt(R);
        Real Rp = 0.5 * s * a + 0.5 * dc_dz * z * z + c_eff * z;
        // R'' = (1/2) s^2 + (1/2) c_eff'' z^2 + 2 c_eff' z + c_eff
        Real Rpp = 0.5 * s * s + 0.5 * d2c_dz2 * z * z
                   + 2.0 * dc_dz * z + c_eff;
        // f' = (1/2) s + R'/(2 sqrt(R))   (linear-in-z half drops out of f'')
        // f'' = R''/(2 r) - R'^2 / (4 r^3)
        return Rpp / (2.0 * r) - (Rp * Rp) / (4.0 * r * R);
    }

    std::vector<Real> K7Shape::dfdParams(
            Real z, const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == 6, "K7Shape: expected 6 params");
        Real s = p[0];
        Real c   = std::max(p[1], 0.0);
        Real cmm = std::max(p[2], 0.0);
        Real cmp = std::max(p[3], 0.0);
        Real cfm = std::max(p[4], 0.0);
        Real cfp = std::max(p[5], 0.0);
        Real a = 1.0 + s * z;
        Real c_eff, dc_dz, d2c_dz2;
        k7CEff(z, c, cmm, cmp, cfm, cfp, c_eff, dc_dz, d2c_dz2);
        Real R = 0.25 * a * a + 0.5 * c_eff * z * z;
        Real sqrtR = (R > 0.0) ? std::sqrt(R) : 1.0;  // guarded; R>0 in fit
        K7Blend b = k7BlendAt(z);

        // f = (1/2) a + sqrt(R), so df/dq = (dR/dq) / (2 sqrt(R)) for any
        // shape param q != s (where the (1/2) a term also contributes).
        // c_eff = c (1 - T1) + c_mid (T1 - T2) + c_far T2
        // dR/dq = (1/2) (dc_eff/dq) z^2

        // d/ds : also contributes via (1/2) a
        Real ds = 0.5 * z + 0.25 * a * z / sqrtR;

        // d/dc
        Real dc = 0.5 * z * z * (1.0 - b.T1) / (2.0 * sqrtR);

        // d/dc_mid_minus, d/dc_mid_plus : (T1 - T2) on respective side
        Real dc_mid_factor = 0.5 * z * z * (b.T1 - b.T2) / (2.0 * sqrtR);
        Real dcmm = (z < 0.0) ? dc_mid_factor : 0.0;
        Real dcmp = (z >= 0.0) ? dc_mid_factor : 0.0;

        // d/dc_far_minus, d/dc_far_plus : T2 on respective side
        Real dc_far_factor = 0.5 * z * z * b.T2 / (2.0 * sqrtR);
        Real dcfm = (z < 0.0) ? dc_far_factor : 0.0;
        Real dcfp = (z >= 0.0) ? dc_far_factor : 0.0;

        return {ds, dc, dcmm, dcmp, dcfm, dcfp};
    }


    // ===== SplShape: quintic B-spline belly + softplus wings =================

    namespace {

        //! log(1+e^{tau u}) / tau, overflow-safe.
        inline Real splSoftplus(Real u, Real tau) {
            Real x = tau * u;
            Real lse = (x > 0.0) ? (x + std::log1p(std::exp(-x)))
                                 : std::log1p(std::exp(x));
            return lse / tau;
        }

        inline Real splSigmoid(Real x) {
            if (x >= 0.0)
                return 1.0 / (1.0 + std::exp(-x));
            Real e = std::exp(x);
            return e / (1.0 + e);
        }

        /*! Cox-de Boor: value and derivatives (to `maxOrder`) of every
            degree-`p` B-spline basis function at z.  `out` is
            (maxOrder+1) x nBasis, row-major. */
        void splBsplineDerivs(const std::vector<Real>& T, Size p, Real z,
                              Size nBasis, Size maxOrder,
                              std::vector<std::vector<Real> >& out) {
            Size nAll = T.size() - 1;
            Real hi = T.back();
            // table[k][q] = k-th derivative of the degree-q bases
            std::vector<std::vector<std::vector<Real> > > table(
                maxOrder + 1, std::vector<std::vector<Real> >(p + 1));

            std::vector<Real> deg0(nAll, 0.0);
            for (Size j = 0; j < nAll; ++j)
                if ((T[j] <= z && z < T[j+1]) ||
                    (z >= hi && T[j] < T[j+1] && T[j+1] >= hi))
                    deg0[j] = 1.0;
            table[0][0] = deg0;
            for (Size q = 1; q <= p; ++q) {
                const std::vector<Real>& prev = table[0][q-1];
                std::vector<Real> cur(nAll - q, 0.0);
                for (Size j = 0; j < cur.size(); ++j) {
                    Real d1 = T[j+q] - T[j];
                    Real d2 = T[j+q+1] - T[j+1];
                    Real a = (d1 > 0.0) ? (z - T[j]) / d1 * prev[j] : 0.0;
                    Real b = (d2 > 0.0) ? (T[j+q+1] - z) / d2 * prev[j+1] : 0.0;
                    cur[j] = a + b;
                }
                table[0][q] = cur;
            }
            // D^k B_{j,q} = q * [ D^{k-1}B_{j,q-1}/(T[j+q]-T[j])
            //                   - D^{k-1}B_{j+1,q-1}/(T[j+q+1]-T[j+1]) ]
            for (Size k = 1; k <= maxOrder; ++k) {
                for (Size q = k; q <= p; ++q) {
                    const std::vector<Real>& src = table[k-1][q-1];
                    std::vector<Real> cur(T.size() - 1 - q, 0.0);
                    for (Size j = 0; j < cur.size(); ++j) {
                        Real d1 = T[j+q] - T[j];
                        Real d2 = T[j+q+1] - T[j+1];
                        Real a = (d1 > 0.0) ? src[j] / d1 : 0.0;
                        Real b = (d2 > 0.0) ? src[j+1] / d2 : 0.0;
                        cur[j] = Real(q) * (a - b);
                    }
                    table[k][q] = cur;
                }
            }
            out.assign(maxOrder + 1, std::vector<Real>(nBasis, 0.0));
            for (Size k = 0; k <= maxOrder; ++k)
                for (Size j = 0; j < nBasis; ++j)
                    out[k][j] = table[k][p][j];
        }

    }  // namespace

    SplShape::SplShape(std::vector<Real> knots, Real switchLeft,
                       Real switchRight, Real tau)
    : knots_(std::move(knots)) {
        Size n = knots_.size();
        QL_REQUIRE(n >= 3, "SplShape: need >= 3 knots, got " << n);
        for (Size i = 1; i < n; ++i)
            QL_REQUIRE(knots_[i] > knots_[i-1],
                       "SplShape: knots must be strictly increasing");
        zl_ = knots_.front();
        zr_ = knots_.back();
        xLeft_  = (switchLeft  == Null<Real>()) ? zl_ : switchLeft;
        xRight_ = (switchRight == Null<Real>()) ? zr_ : switchRight;

        QL_REQUIRE(tau > 0.0, "SplShape: tau must be > 0, got " << tau);
        tau_ = tau;
        // The wing ramp sharpness is CONFIG, not a fitted channel.  A small tau
        // smears the softplus all the way to the money where it competes with
        // the belly and destroys the f(0)=1 pin -- fatal at short maturities
        // where the Lee bound permits very large wing slopes.
        Real leak = splSoftplus(xLeft_, tau_) + splSoftplus(-xRight_, tau_);
        QL_REQUIRE(leak <= SPL_MAX_ATM_LEAK,
                   "SplShape: tau=" << tau_ << " leaks " << leak
                   << " of each unit of wing slope into the money (switch points "
                   << xLeft_ << "/" << xRight_ << "); max " << SPL_MAX_ATM_LEAK);

        // Clamped knot vector: end breakpoints repeated p times extra.
        //
        // Degree 5, not 3.  A cubic basis is only C2 across simple interior
        // knots, so the THIRD derivative jumps at every knot.  Dupire local
        // vol differentiates total variance twice in strike, and those jumps
        // were measured at 360-441x the between-knot curvature background --
        // ridges in the LV surface, which gamma and vanna then differentiate
        // again.  Quintic is C4, so the third and fourth derivatives are
        // continuous.  Raising the DEGREE rather than abandoning the B-spline
        // keeps positivity by construction (see the class docs).
        Size p = SPL_BSPLINE_DEGREE;
        std::vector<Real> T;
        for (Size i = 0; i < p; ++i) T.push_back(knots_.front());
        for (Size i = 0; i < n; ++i)  T.push_back(knots_[i]);
        for (Size i = 0; i < p; ++i) T.push_back(knots_.back());
        Size nBasis = T.size() - p - 1;
        QL_REQUIRE(nBasis >= 6,
                   "SplShape: knot grid too coarse -- " << n << " knots give only "
                   << nBasis << " basis functions, need >= 6");

        // Tie the first three and last three control points so base' = base'' = 0
        // at the outer knots: beyond them the belly is CONSTANT, so this makes
        // that join C2 and leaves the wings owning all asymptotic slope.  Each
        // tied group is a SUM of non-negative bases, so non-negativity and
        // partition-of-unity (hence positivity) survive the reduction.
        nFree_ = nBasis - 4;
        std::vector<std::vector<Real> > E(nBasis, std::vector<Real>(nFree_, 0.0));
        for (Size i = 0; i < 3; ++i)          E[i][0] = 1.0;
        for (Size i = 1; i + 1 < nFree_; ++i) E[i+2][i] = 1.0;
        for (Size i = 0; i < 3; ++i)          E[nBasis-3+i][nFree_-1] = 1.0;

        // Collapse to per-interval POWER coefficients about interval midpoints.
        // Running Cox-de Boor per quote is correct but far too slow in the fit
        // hot loop; this turns evaluation into a Horner on SPL_N_POW numbers.
        Size nInt = n - 1;
        mids_.resize(nInt);
        cmap_.assign(nInt, std::vector<std::vector<Real> >(
                               SPL_N_POW, std::vector<Real>(nFree_, 0.0)));
        const Real fact[SPL_N_POW] = { 1.0, 1.0, 2.0, 6.0, 24.0, 120.0 };
        std::vector<std::vector<Real> > d;
        for (Size i = 0; i < nInt; ++i) {
            mids_[i] = 0.5 * (knots_[i] + knots_[i+1]);
            splBsplineDerivs(T, p, mids_[i], nBasis, SPL_BSPLINE_DEGREE, d);
            for (Size k = 0; k < SPL_N_POW; ++k)
                for (Size j = 0; j < nFree_; ++j) {
                    Real acc = 0.0;
                    for (Size bb = 0; bb < nBasis; ++bb)
                        acc += d[k][bb] * E[bb][j];
                    cmap_[i][k][j] = acc / fact[k];
                }
        }

        // ATM level by NORMALISATION, not by pinning one control point:
        //     base(z) = (1 - wings(0)) * S(z)/S(0),  S(z) = sum_j q_j phi_j(z)
        // so f(0) = 1 identically for any positive control points.  Pinning a
        // single control point has a reachable infeasible region -- its own
        // influence phi_j(0) is at most ~2/3, so against large neighbours it has
        // to go negative and the shape must refuse an ordinary parameter vector.
        // q_0 is fixed at 1: base is invariant under a common rescaling of q, so
        // leaving every control point free would give the optimizer a flat
        // direction.
        phi0_ = phiAt(0.0, 0);
        nNode_ = nFree_ - 1;
    }

    Size SplShape::intervalAt(Real z) const {
        Size i = Size(std::upper_bound(knots_.begin(), knots_.end(), z)
                      - knots_.begin());
        i = (i == 0) ? 0 : i - 1;
        if (i + 1 >= knots_.size()) i = knots_.size() - 2;
        return i;
    }

    std::vector<Real> SplShape::phiAt(Real z, Size order) const {
        Size i = intervalAt(z);
        Real u = z - mids_[i];
        const std::vector<std::vector<Real> >& c = cmap_[i];
        std::vector<Real> out(nFree_, 0.0);
        for (Size j = 0; j < nFree_; ++j) {
            switch (order) {
              case 0:
                out[j] = c[0][j] + u*(c[1][j] + u*(c[2][j] + u*(c[3][j]
                       + u*(c[4][j] + u*c[5][j])))); break;
              case 1:
                out[j] = c[1][j] + u*(2.0*c[2][j] + u*(3.0*c[3][j]
                       + u*(4.0*c[4][j] + u*5.0*c[5][j]))); break;
              case 2:
                out[j] = 2.0*c[2][j] + u*(6.0*c[3][j] + u*(12.0*c[4][j]
                       + u*20.0*c[5][j])); break;
              case 3:
                out[j] = 6.0*c[3][j] + u*(24.0*c[4][j] + u*60.0*c[5][j]); break;
              case 4:
                out[j] = 24.0*c[4][j] + u*120.0*c[5][j]; break;
              default:
                out[j] = 120.0*c[5][j]; break;
            }
        }
        return out;
    }

    void SplShape::ensureCoef(const std::vector<Real>& p) const {
        QL_REQUIRE(p.size() == nNode_ + 2,
                   "SplShape: expected " << (nNode_ + 2) << " params ("
                   << nNode_ << " belly control points + sL, sR), got " << p.size());
        if (cacheValid_ && p == cacheKey_)
            return;

        Real sL = std::max(p[nNode_], 0.0);
        Real sR = std::max(p[nNode_ + 1], 0.0);

        std::vector<Real> q(nFree_, 0.0);
        q[0] = 1.0;   // scale anchor (see the constructor)
        // Clamp to a positive floor rather than rejecting: chloride log-transforms
        // these channels so the true params are positive, but the optimizer's
        // finite-difference probe lands the raw value a hair below zero.  Same
        // tolerance K5Shape applies to its curvature params, for the same reason.
        for (Size i = 1; i < nFree_; ++i)
            q[i] = std::max(p[i-1], SPL_COEF_FLOOR);

        Real wings0 = sL * splSoftplus(xLeft_, tau_)
                    + sR * splSoftplus(-xRight_, tau_);
        QL_REQUIRE(wings0 < 1.0,
                   "SplShape: wings consume the ATM level (wings(0)=" << wings0
                   << " >= 1); sL=" << sL << " sR=" << sR << " tau=" << tau_);
        Real s0 = 0.0;
        for (Size j = 0; j < nFree_; ++j) s0 += phi0_[j] * q[j];
        Real scale = (1.0 - wings0) / s0;

        Size nInt = mids_.size();
        poly_.assign(nInt, std::vector<Real>(SPL_N_POW, 0.0));
        for (Size i = 0; i < nInt; ++i)
            for (Size k = 0; k < SPL_N_POW; ++k) {
                Real acc = 0.0;
                for (Size j = 0; j < nFree_; ++j)
                    acc += cmap_[i][k][j] * q[j];
                poly_[i][k] = acc * scale;
            }
        coef_ = q;
        sLc_ = sL;
        sRc_ = sR;
        scale_ = scale;
        wings0_ = wings0;
        cacheKey_ = p;
        cacheValid_ = true;
    }

    Real SplShape::bellyEval(Real z, Size order) const {
        if (z < zl_ || z > zr_) {
            if (order > 0) return 0.0;
            z = (z < zl_) ? zl_ : zr_;
        }
        Size i = intervalAt(z);
        Real u = z - mids_[i];
        const std::vector<Real>& a = poly_[i];
        switch (order) {
          case 0: return a[0] + u*(a[1] + u*(a[2] + u*(a[3] + u*(a[4] + u*a[5]))));
          case 1: return a[1] + u*(2.0*a[2] + u*(3.0*a[3] + u*(4.0*a[4] + u*5.0*a[5])));
          case 2: return 2.0*a[2] + u*(6.0*a[3] + u*(12.0*a[4] + u*20.0*a[5]));
          case 3: return 6.0*a[3] + u*(24.0*a[4] + u*60.0*a[5]);
          case 4: return 24.0*a[4] + u*120.0*a[5];
          default: return 120.0*a[5];
        }
    }

    Real SplShape::f(Real z, const std::vector<Real>& p) const {
        ensureCoef(p);
        return bellyEval(z, 0)
             + sLc_ * splSoftplus(xLeft_ - z, tau_)
             + sRc_ * splSoftplus(z - xRight_, tau_);
    }

    Real SplShape::dfdz(Real z, const std::vector<Real>& p) const {
        ensureCoef(p);
        return bellyEval(z, 1)
             - sLc_ * splSigmoid(tau_ * (xLeft_ - z))
             + sRc_ * splSigmoid(tau_ * (z - xRight_));
    }

    Real SplShape::d2fdz2(Real z, const std::vector<Real>& p, Real) const {
        // Analytic.  The base class would central-difference dfdz, and
        // differencing across a knot re-introduces an artifact right where the
        // belly is stitched.
        ensureCoef(p);
        Real gL = splSigmoid(tau_ * (xLeft_ - z));
        Real gR = splSigmoid(tau_ * (z - xRight_));
        return bellyEval(z, 2)
             + sLc_ * tau_ * gL * (1.0 - gL)
             + sRc_ * tau_ * gR * (1.0 - gR);
    }

    std::vector<Real> SplShape::dfdParams(
            Real z, const std::vector<Real>& p) const {
        ensureCoef(p);
        std::vector<Real> out(nNode_ + 2, 0.0);

        // base(z) = (1-W) S(z)/S(0), so a control point moves the belly directly
        // AND through the normalisation:
        //     d base / d q_j = scale * [ phi_j(z) - (S(z)/S(0)) phi_j(0) ]
        Real zc = (z < zl_) ? zl_ : ((z > zr_) ? zr_ : z);
        std::vector<Real> phi = phiAt(zc, 0);
        Real ratio = bellyEval(z, 0) / (1.0 - wings0_);
        for (Size j = 1; j < nFree_; ++j)
            out[j-1] = scale_ * (phi[j] - ratio * phi0_[j]);
        // Wing slopes: direct term, less what the normalisation takes back.
        out[nNode_]     = splSoftplus(xLeft_ - z, tau_)
                          - ratio * splSoftplus(xLeft_, tau_);
        out[nNode_ + 1] = splSoftplus(z - xRight_, tau_)
                          - ratio * splSoftplus(-xRight_, tau_);
        return out;
    }


    // ===== ParametricVolTermStructure =========================================

    namespace {

        std::vector<Time> datesToTimes(const Date& refDate,
                                       const std::vector<Date>& dates,
                                       const DayCounter& dc) {
            std::vector<Time> times(dates.size());
            for (Size i = 0; i < dates.size(); ++i) {
                times[i] = dc.yearFraction(refDate, dates[i]);
                QL_REQUIRE(times[i] > 0.0,
                           "date[" << i << "] (" << dates[i]
                           << ") must be after reference date ("
                           << refDate << ")");
                if (i > 0)
                    QL_REQUIRE(times[i] > times[i-1],
                               "dates must be strictly increasing");
            }
            return times;
        }

        void validateSlices(const std::vector<Time>& T,
                            const std::vector<ParametricVolSlice>& slices) {
            QL_REQUIRE(T.size() == slices.size(),
                       "dates size (" << T.size()
                       << ") must match slices size (" << slices.size() << ")");
            QL_REQUIRE(!slices.empty(), "at least one slice required");
            for (Size i = 0; i < slices.size(); ++i) {
                QL_REQUIRE(slices[i].atmIv > 0.0,
                           "slice[" << i << "] atmIv must be > 0");
            }
        }

    }  // namespace

    ParametricVolTermStructure::ParametricVolTermStructure(
            const Date& referenceDate,
            const std::vector<Date>& dates,
            const std::vector<ParametricVolSlice>& slices,
            ext::shared_ptr<ParametricVolShape> shape,
            Handle<Quote> spot,
            Handle<YieldTermStructure> riskFreeRate,
            Handle<YieldTermStructure> dividendYield,
            const DayCounter& dc)
        : BlackVolatilityTermStructure(referenceDate, Calendar(), Following, dc),
          T_(datesToTimes(referenceDate, dates, dc)),
          slices_(slices),
          shape_(std::move(shape)),
          spot_(std::move(spot)),
          riskFreeRate_(std::move(riskFreeRate)),
          dividendYield_(std::move(dividendYield)),
          dividends_()
    {
        validateSlices(T_, slices_);
        QL_REQUIRE(shape_, "shape must not be null");
        QL_REQUIRE(!spot_.empty(), "spot handle must not be empty");
        QL_REQUIRE(!riskFreeRate_.empty(), "risk-free rate handle must not be empty");
        QL_REQUIRE(!dividendYield_.empty(), "dividend yield handle must not be empty");
        registerWith(spot_);
        registerWith(riskFreeRate_);
        registerWith(dividendYield_);
    }

    ParametricVolTermStructure::ParametricVolTermStructure(
            const Date& referenceDate,
            const std::vector<Date>& dates,
            const std::vector<ParametricVolSlice>& slices,
            ext::shared_ptr<ParametricVolShape> shape,
            Handle<Quote> spot,
            Handle<YieldTermStructure> riskFreeRate,
            Handle<YieldTermStructure> dividendYield,
            DividendSchedule dividends,
            const DayCounter& dc)
        : BlackVolatilityTermStructure(referenceDate, Calendar(), Following, dc),
          T_(datesToTimes(referenceDate, dates, dc)),
          slices_(slices),
          shape_(std::move(shape)),
          spot_(std::move(spot)),
          riskFreeRate_(std::move(riskFreeRate)),
          dividendYield_(std::move(dividendYield)),
          dividends_(std::move(dividends))
    {
        validateSlices(T_, slices_);
        QL_REQUIRE(shape_, "shape must not be null");
        QL_REQUIRE(!spot_.empty(), "spot handle must not be empty");
        QL_REQUIRE(!riskFreeRate_.empty(), "risk-free rate handle must not be empty");
        QL_REQUIRE(!dividendYield_.empty(), "dividend yield handle must not be empty");
        registerWith(spot_);
        registerWith(riskFreeRate_);
        registerWith(dividendYield_);
    }

    Real ParametricVolTermStructure::forward(Time t) const {
        Real S = spot_->value();
        Real df = riskFreeRate_->discount(t);
        Real dq = dividendYield_->discount(t);

        Real pvDivs = 0.0;
        if (!dividends_.empty()) {
            Date refDate = referenceDate();
            DayCounter dc = dayCounter();
            for (const auto& div : dividends_) {
                Date exDate = div->date();
                if (exDate <= refDate)
                    continue;
                Time tDiv = dc.yearFraction(refDate, exDate);
                if (tDiv < t) {
                    pvDivs += div->amount() * riskFreeRate_->discount(tDiv);
                }
            }
        }
        return (S - pvDivs) * dq / df;
    }

    Real ParametricVolTermStructure::z(Size i, Real k) const {
        const auto& s = slices_.at(i);
        Real sigmaHat = s.atmIv * std::sqrt(T_[i]);
        return k / sigmaHat;
    }

    Real ParametricVolTermStructure::totalVariance(Size i, Real k) const {
        const auto& s = slices_.at(i);
        Real sigmaHat = s.atmIv * std::sqrt(T_[i]);
        Real zv = k / sigmaHat;
        Real fv = shape_->f(zv, s.params);
        // w = T * atm_iv^2 * f(z)  =  sigmaHat^2 * f(z)
        return sigmaHat * sigmaHat * fv;
    }

    Real ParametricVolTermStructure::totalVarianceStrikeDerivative(
            Size i, Real k) const {
        const auto& s = slices_.at(i);
        Real sigmaHat = s.atmIv * std::sqrt(T_[i]);
        Real zv = k / sigmaHat;
        Real dfdz = shape_->dfdz(zv, s.params);
        // dw/dk = sigmaHat^2 * df/dk = sigmaHat^2 * df/dz * dz/dk
        //       = sigmaHat^2 * df/dz / sigmaHat = sigmaHat * df/dz
        return sigmaHat * dfdz;
    }

    ParametricVolTermStructure::TimeWeights
    ParametricVolTermStructure::timeWeights(Time t) const {
        const Size N = T_.size();
        TimeWeights tw{};

        if (t <= T_.front()) {
            tw.sLo = 0;
            tw.sHi = 0;
            tw.wLo = t / T_.front();
            tw.wHi = 0.0;
            tw.leftExtrap = true;
            return tw;
        }
        tw.leftExtrap = false;

        if (N == 1 || t >= T_.back()) {
            if (N == 1) {
                tw.sLo = 0;
                tw.sHi = 0;
                tw.wLo = 1.0;
                tw.wHi = 0.0;
                return tw;
            }
            tw.sLo = N - 2;
            tw.sHi = N - 1;
            Time dT = T_[N-1] - T_[N-2];
            Real a = (t - T_[N-2]) / dT;
            tw.wLo = 1.0 - a;
            tw.wHi = a;
            return tw;
        }

        auto it = std::upper_bound(T_.begin(), T_.end(), t);
        Size hi = static_cast<Size>(it - T_.begin());
        Size lo = hi - 1;
        Time dT = T_[hi] - T_[lo];
        Real a = (t - T_[lo]) / dT;
        tw.sLo = lo;
        tw.sHi = hi;
        tw.wLo = 1.0 - a;
        tw.wHi = a;
        return tw;
    }

    Real ParametricVolTermStructure::totalVariance(Real k, Time t) const {
        QL_REQUIRE(t > 0.0, "time must be > 0");
        TimeWeights tw = timeWeights(t);
        Real w_lo = totalVariance(tw.sLo, k);
        if (tw.leftExtrap || tw.sHi == tw.sLo)
            return tw.wLo * w_lo;
        Real w_hi = totalVariance(tw.sHi, k);
        return tw.wLo * w_lo + tw.wHi * w_hi;
    }

    Real ParametricVolTermStructure::totalVarianceStrikeDerivative(
            Real k, Time t) const {
        QL_REQUIRE(t > 0.0, "time must be > 0");
        TimeWeights tw = timeWeights(t);
        Real d_lo = totalVarianceStrikeDerivative(tw.sLo, k);
        if (tw.leftExtrap || tw.sHi == tw.sLo)
            return tw.wLo * d_lo;
        Real d_hi = totalVarianceStrikeDerivative(tw.sHi, k);
        return tw.wLo * d_lo + tw.wHi * d_hi;
    }

    Real ParametricVolTermStructure::totalVarianceStrikeSecondDerivative(
            Size i, Real k) const {
        // w(T,k) = sigmaHat² · f(z),  z = k/sigmaHat,  dz/dk = 1/sigmaHat
        // ⇒ d²w/dk² = sigmaHat² · f''(z) · (1/sigmaHat)² = f''(z)
        const auto& s = slices_.at(i);
        Real sigmaHat = s.atmIv * std::sqrt(T_[i]);
        Real zv = k / sigmaHat;
        return shape_->d2fdz2(zv, s.params);
    }

    Real ParametricVolTermStructure::totalVarianceStrikeSecondDerivative(
            Real k, Time t) const {
        QL_REQUIRE(t > 0.0, "time must be > 0");
        TimeWeights tw = timeWeights(t);
        Real d_lo = totalVarianceStrikeSecondDerivative(tw.sLo, k);
        if (tw.leftExtrap || tw.sHi == tw.sLo)
            return tw.wLo * d_lo;
        Real d_hi = totalVarianceStrikeSecondDerivative(tw.sHi, k);
        return tw.wLo * d_lo + tw.wHi * d_hi;
    }

    Real ParametricVolTermStructure::totalVarianceTimeDerivative(
            Real k, Time t) const {
        QL_REQUIRE(t > 0.0, "time must be > 0");
        const Size N = T_.size();
        QL_REQUIRE(N >= 2,
                   "totalVarianceTimeDerivative: needs >=2 pillars "
                   "(local vol is degenerate on a single-pillar surface)");

        // Linear-in-w time interpolation gives a piecewise-constant slope
        // per pillar interval.  We use the right-continuous convention:
        // for t exactly on a pillar, return the slope of the interval
        // starting at that pillar.
        Size lo, hi;
        if (t <= T_.front()) {
            // Left extrap: w(t,k) = (t/T_0)·w_0(k)  ⇒  dw/dT = w_0(k)/T_0
            return totalVariance(Size(0), k) / T_.front();
        } else if (t >= T_.back()) {
            // Right extrap: hold slope of last interior interval.
            lo = N - 2;
            hi = N - 1;
        } else {
            auto it = std::upper_bound(T_.begin(), T_.end(), t);
            hi = static_cast<Size>(it - T_.begin());
            lo = hi - 1;
        }
        Real w_lo = totalVariance(lo, k);
        Real w_hi = totalVariance(hi, k);
        return (w_hi - w_lo) / (T_[hi] - T_[lo]);
    }

    Real ParametricVolTermStructure::localVariance(Real k, Time t) const {
        QL_REQUIRE(t > 0.0, "localVariance: time must be > 0");

        Real w     = totalVariance(k, t);
        Real dwdk  = totalVarianceStrikeDerivative(k, t);
        Real d2wdk = totalVarianceStrikeSecondDerivative(k, t);
        Real dwdt  = totalVarianceTimeDerivative(k, t);

        // Flat-vol degenerate case: only the time derivative survives.
        if (dwdk == 0.0 && d2wdk == 0.0)
            return dwdt;

        QL_REQUIRE(w > 0.0,
                   "localVariance: non-positive total variance " << w
                   << " at k=" << k << ", t=" << t);

        // Gatheral / Dupire denominator in y = log(K/F):
        //   D = 1 − k/w·dw/dk + ¼(−¼ − 1/w + k²/w²)·(dw/dk)² + ½·d²w/dk²
        Real den1 = 1.0 - k / w * dwdk;
        Real den2 = 0.25 * (-0.25 - 1.0 / w + k * k / (w * w)) * dwdk * dwdk;
        Real den3 = 0.5 * d2wdk;
        Real den  = den1 + den2 + den3;

        QL_REQUIRE(den > 0.0,
                   "ParametricVolTermStructure::localVariance: non-positive "
                   "Dupire denominator " << den << " at k=" << k
                   << ", t=" << t << " (calendar/butterfly arb in surface)");
        return dwdt / den;
    }

    Size ParametricVolTermStructure::localVarianceSlice(
            Time t, const Real* k, Size n, Real* out) const {
        QL_REQUIRE(t > 0.0, "localVarianceSlice: time must be > 0");
        const Size N = T_.size();
        if (N < 2) {
            for (Size j = 0; j < n; ++j)
                out[j] = Null<Real>();
            return n;
        }

        // For every t the pillars timeWeights() brackets are also the
        // interval totalVarianceTimeDerivative() takes the slope of (left
        // extrapolation excepted), so w at each pillar serves both.
        const TimeWeights tw = timeWeights(t);
        const ParametricVolSlice& sLo = slices_.at(tw.sLo);
        const Real hatLo = sLo.atmIv * std::sqrt(T_[tw.sLo]);
        const bool oneSlice = tw.leftExtrap || tw.sHi == tw.sLo;
        const ParametricVolSlice& sHi = slices_.at(tw.sHi);
        const Real hatHi = sHi.atmIv * std::sqrt(T_[tw.sHi]);
        const Time dT = T_[tw.sHi] - T_[tw.sLo];

        Size nIllegal = 0;
        for (Size j = 0; j < n; ++j) {
            const Real kj = k[j];
            // Same expressions, in the same order, as totalVariance(i, k),
            // totalVarianceStrikeDerivative(i, k) and
            // totalVarianceStrikeSecondDerivative(i, k).
            const Real zLo = kj / hatLo;
            const Real wLo = hatLo * hatLo * shape_->f(zLo, sLo.params);
            const Real dLo = hatLo * shape_->dfdz(zLo, sLo.params);
            const Real d2Lo = shape_->d2fdz2(zLo, sLo.params);

            Real w, dwdk, d2wdk, dwdt;
            if (oneSlice) {
                w = tw.wLo * wLo;
                dwdk = tw.wLo * dLo;
                d2wdk = tw.wLo * d2Lo;
                dwdt = wLo / T_.front();
            } else {
                const Real zHi = kj / hatHi;
                const Real wHi = hatHi * hatHi * shape_->f(zHi, sHi.params);
                const Real dHi = hatHi * shape_->dfdz(zHi, sHi.params);
                const Real d2Hi = shape_->d2fdz2(zHi, sHi.params);
                w = tw.wLo * wLo + tw.wHi * wHi;
                dwdk = tw.wLo * dLo + tw.wHi * dHi;
                d2wdk = tw.wLo * d2Lo + tw.wHi * d2Hi;
                dwdt = (wHi - wLo) / dT;
            }

            // localVariance(), with its QL_REQUIREs and localVol()'s
            // non-negativity check turned into Null<Real>().
            Real lv2;
            if (dwdk == 0.0 && d2wdk == 0.0) {
                lv2 = dwdt;
            } else if (!(w > 0.0)) {
                lv2 = Null<Real>();
            } else {
                const Real den1 = 1.0 - kj / w * dwdk;
                const Real den2 = 0.25 * (-0.25 - 1.0 / w + kj * kj / (w * w))
                                  * dwdk * dwdk;
                const Real den3 = 0.5 * d2wdk;
                const Real den = den1 + den2 + den3;
                lv2 = (den > 0.0) ? dwdt / den : Null<Real>();
            }
            if (lv2 == Null<Real>() || !(lv2 >= 0.0)) {
                out[j] = Null<Real>();
                ++nIllegal;
            } else {
                out[j] = lv2;
            }
        }
        return nIllegal;
    }

    Real ParametricVolTermStructure::localVol(Real k, Time t) const {
        Real lv2 = localVariance(k, t);
        QL_REQUIRE(lv2 >= 0.0,
                   "ParametricVolTermStructure::localVol: negative local "
                   "variance " << lv2 << " at k=" << k << ", t=" << t);
        return std::sqrt(lv2);
    }

    Volatility ParametricVolTermStructure::blackVolImpl(Time t,
                                                        Real strike) const {
        if (t < 1e-14) t = 1e-14;
        Real F = forward(t);
        Real k = std::log(strike / F);
        Real w = totalVariance(k, t);
        QL_REQUIRE(w >= 0.0, "negative total variance at (k=" << k
                              << ", t=" << t << ")");
        return std::sqrt(w / t);
    }

    void ParametricVolTermStructure::setSlice(Size i,
                                              Real atmIv,
                                              const std::vector<Real>& params) {
        QL_REQUIRE(i < slices_.size(), "slice index out of range");
        QL_REQUIRE(atmIv > 0.0, "atmIv must be > 0");
        slices_[i].atmIv = atmIv;
        slices_[i].params = params;
        notifyObservers();
    }

    void ParametricVolTermStructure::setSlices(
            const std::vector<ParametricVolSlice>& slices) {
        QL_REQUIRE(slices.size() == T_.size(),
                   "slices size must match pillar count");
        for (Size i = 0; i < slices.size(); ++i)
            QL_REQUIRE(slices[i].atmIv > 0.0,
                       "slice[" << i << "] atmIv must be > 0");
        slices_ = slices;
        notifyObservers();
    }

    // ----- Batch evaluation ---------------------------------------------------

    std::vector<Real> ParametricVolTermStructure::batchBlackVol(
            const std::vector<Size>& sliceIndices,
            const std::vector<Real>& strikes) const {
        QL_REQUIRE(sliceIndices.size() == strikes.size(),
                   "sliceIndices size (" << sliceIndices.size()
                   << ") must match strikes size (" << strikes.size() << ")");
        std::vector<Real> out(strikes.size());
        for (Size j = 0; j < strikes.size(); ++j) {
            Size i = sliceIndices[j];
            QL_REQUIRE(i < slices_.size(),
                       "sliceIndices[" << j << "] = " << i
                       << " out of range (N=" << slices_.size() << ")");
            const auto& s = slices_[i];
            Real T = T_[i];
            Real F = forward(T);
            Real k = std::log(strikes[j] / F);
            Real sigmaHat = s.atmIv * std::sqrt(T);
            Real zv = k / sigmaHat;
            Real w = sigmaHat * sigmaHat * shape_->f(zv, s.params);
            QL_REQUIRE(w >= 0.0, "negative total variance at slice " << i);
            out[j] = std::sqrt(w / T);
        }
        return out;
    }

    std::vector<Real> ParametricVolTermStructure::batchBlackVolAtTimes(
            const std::vector<Real>& times,
            const std::vector<Real>& strikes) const {
        QL_REQUIRE(times.size() == strikes.size(),
                   "times size (" << times.size()
                   << ") must match strikes size (" << strikes.size() << ")");
        std::vector<Real> out(times.size());
        for (Size j = 0; j < times.size(); ++j) {
            Time t = times[j];
            if (t < 1e-14) t = 1e-14;
            Real F = forward(t);
            Real k = std::log(strikes[j] / F);
            Real w = totalVariance(k, t);
            QL_REQUIRE(w >= 0.0, "negative total variance at t=" << t);
            out[j] = std::sqrt(w / t);
        }
        return out;
    }

    std::vector<Real> ParametricVolTermStructure::batchImpliedVolGradient(
            const std::vector<Size>& sliceIndices,
            const std::vector<Real>& strikes) const {
        QL_REQUIRE(sliceIndices.size() == strikes.size(),
                   "sliceIndices size (" << sliceIndices.size()
                   << ") must match strikes size (" << strikes.size() << ")");
        QL_REQUIRE(!slices_.empty(), "no slices");
        const Size n_shape = slices_.front().params.size();
        const Size stride = 1 + n_shape;
        std::vector<Real> out(strikes.size() * stride);
        for (Size j = 0; j < strikes.size(); ++j) {
            Size i = sliceIndices[j];
            QL_REQUIRE(i < slices_.size(),
                       "sliceIndices[" << j << "] = " << i
                       << " out of range");
            const auto& s = slices_[i];
            QL_REQUIRE(s.params.size() == n_shape,
                       "slice " << i << " has " << s.params.size()
                       << " params, expected " << n_shape);
            Real T = T_[i];
            Real F = forward(T);
            Real k = std::log(strikes[j] / F);
            Real atm = s.atmIv;
            Real sigmaHat = atm * std::sqrt(T);
            Real zv = k / sigmaHat;
            Real fv = shape_->f(zv, s.params);
            QL_REQUIRE(fv > 0.0, "f(z) <= 0 at slice " << i << ", z=" << zv);
            Real sqrtF = std::sqrt(fv);
            Real dfdz1 = shape_->dfdz(zv, s.params);
            // ∂σ/∂atm = (2f - z·f') / (2·√f)
            out[j * stride + 0] = (2.0 * fv - zv * dfdz1) / (2.0 * sqrtF);
            // ∂σ/∂params_p = atm · ∂f/∂params_p / (2·√f)
            auto dfdp = shape_->dfdParams(zv, s.params);
            QL_REQUIRE(dfdp.size() == n_shape,
                       "shape returned " << dfdp.size()
                       << " grads, expected " << n_shape);
            Real scale = atm / (2.0 * sqrtF);
            for (Size p = 0; p < n_shape; ++p)
                out[j * stride + 1 + p] = scale * dfdp[p];
        }
        return out;
    }

    // ----- Arbitrage checks (Klassen 2017 §3) ---------------------------------

    Real ParametricVolTermStructure::butterflyDensity(Size i,
                                                      Real z,
                                                      Real h) const {
        const auto& s = slices_.at(i);
        Real sigmaHat = s.atmIv * std::sqrt(T_[i]);
        Real fv  = shape_->f(z, s.params);
        QL_REQUIRE(fv > 0.0, "f(z) <= 0 at z=" << z << " (pillar " << i << ")");
        Real dfdz1   = shape_->dfdz(z, s.params);
        Real d2fdz2v = shape_->d2fdz2(z, s.params, h);
        Real term1 = 1.0 - z * dfdz1 / (2.0 * fv);
        Real g = term1 * term1
                 - 0.25 * dfdz1 * dfdz1 / fv
                 - (sigmaHat * sigmaHat / 16.0) * dfdz1 * dfdz1
                 + 0.5 * d2fdz2v;
        return g;
    }

    Real ParametricVolTermStructure::butterflyArbViolation(
            Size i,
            const std::vector<Real>& zGrid,
            Real h) const {
        Real minG = std::numeric_limits<Real>::infinity();
        for (Real z : zGrid) {
            Real g = butterflyDensity(i, z, h);
            if (g < minG) minG = g;
        }
        return std::max(0.0, -minG);
    }

    Real ParametricVolTermStructure::calendarArbViolation(
            const std::vector<Real>& kGrid) const {
        Real worst = 0.0;
        for (Size i = 0; i + 1 < slices_.size(); ++i) {
            for (Real k : kGrid) {
                Real wLo = totalVariance(i, k);
                Real wHi = totalVariance(i + 1, k);
                Real def = wLo - wHi;  // > 0 means calendar arb
                if (def > worst) worst = def;
            }
        }
        return worst;
    }

    // ----- Constraint-grid evaluators (for SLSQP / trust-constr) --------------

    std::vector<Real> ParametricVolTermStructure::butterflyDensityGrid(
            const std::vector<Real>& zGrid, Real h) const {
        const Size N = slices_.size();
        const Size n_z = zGrid.size();
        std::vector<Real> out(N * n_z);
        for (Size i = 0; i < N; ++i) {
            for (Size j = 0; j < n_z; ++j) {
                out[i * n_z + j] = butterflyDensity(i, zGrid[j], h);
            }
        }
        return out;
    }

    namespace {
        // Stateless g(z) on a free-standing slice spec — used to FD-bump
        // params/atm_iv for the gradient without mutating the stored
        // surface slices.
        inline Real gValue(const ParametricVolShape& shape,
                           Real atmIv, Real T,
                           const std::vector<Real>& params,
                           Real z, Real h) {
            Real sigmaHat = atmIv * std::sqrt(T);
            Real fv = shape.f(z, params);
            QL_REQUIRE(fv > 0.0, "f(z) <= 0 in butterfly gradient");
            Real dfdz1 = shape.dfdz(z, params);
            Real d2fdz2v = shape.d2fdz2(z, params, h);
            Real term1 = 1.0 - z * dfdz1 / (2.0 * fv);
            return term1 * term1
                   - 0.25 * dfdz1 * dfdz1 / fv
                   - (sigmaHat * sigmaHat / 16.0) * dfdz1 * dfdz1
                   + 0.5 * d2fdz2v;
        }
    }

    std::vector<Real>
    ParametricVolTermStructure::butterflyDensityGridGradient(
            const std::vector<Real>& zGrid, Real h) const {
        QL_REQUIRE(!slices_.empty(), "no slices");
        const Size N = slices_.size();
        const Size n_z = zGrid.size();
        const Size n_shape = slices_.front().params.size();
        const Size stride = 1 + n_shape;
        std::vector<Real> out(N * n_z * stride);

        for (Size i = 0; i < N; ++i) {
            const auto& s = slices_[i];
            Real T = T_[i];
            std::vector<Real> params_buf = s.params;  // local mutable copy

            for (Size j = 0; j < n_z; ++j) {
                Real z = zGrid[j];

                // ∂g/∂atm_iv via central FD.
                Real h_a = std::max(s.atmIv * 1e-5, 1e-7);
                Real gp_a = gValue(*shape_, s.atmIv + h_a, T, params_buf, z, h);
                Real gm_a = gValue(*shape_, s.atmIv - h_a, T, params_buf, z, h);
                out[(i * n_z + j) * stride + 0] = (gp_a - gm_a) / (2.0 * h_a);

                // ∂g/∂params[p] via central FD on the copied params buffer.
                for (Size p = 0; p < n_shape; ++p) {
                    Real saved = params_buf[p];
                    Real h_p = std::max(std::abs(saved) * 1e-5, 1e-7);
                    params_buf[p] = saved + h_p;
                    Real gp_p = gValue(*shape_, s.atmIv, T, params_buf, z, h);
                    params_buf[p] = saved - h_p;
                    Real gm_p = gValue(*shape_, s.atmIv, T, params_buf, z, h);
                    params_buf[p] = saved;
                    out[(i * n_z + j) * stride + 1 + p] =
                        (gp_p - gm_p) / (2.0 * h_p);
                }
            }
        }
        return out;
    }

    std::vector<Real> ParametricVolTermStructure::calendarDeficitGrid(
            const std::vector<Real>& kGrid) const {
        const Size N = slices_.size();
        if (N < 2) return {};
        const Size n_k = kGrid.size();
        std::vector<Real> out((N - 1) * n_k);
        for (Size i = 1; i < N; ++i) {
            for (Size j = 0; j < n_k; ++j) {
                Real w_lo = totalVariance(i - 1, kGrid[j]);
                Real w_hi = totalVariance(i, kGrid[j]);
                // Deficit ≥ 0  iff w_i ≥ w_{i-1}  iff calendar-arb-free.
                out[(i - 1) * n_k + j] = w_hi - w_lo;
            }
        }
        return out;
    }

    std::vector<Real>
    ParametricVolTermStructure::calendarDeficitGridGradient(
            const std::vector<Real>& kGrid) const {
        QL_REQUIRE(!slices_.empty(), "no slices");
        const Size N = slices_.size();
        if (N < 2) return {};
        const Size n_k = kGrid.size();
        const Size n_shape = slices_.front().params.size();
        const Size grad_per_slice = 1 + n_shape;
        // Row layout for each (pair_idx, k):
        //   [grad_w_lower (atm + params), grad_w_upper (atm + params)]
        const Size row = 2 * grad_per_slice;
        std::vector<Real> out((N - 1) * n_k * row);

        // ∂w/∂atm_iv = 2·atm_iv·T·f(z)  +  atm_iv²·T·f'(z)·∂z/∂atm_iv
        // where z = k/(atm_iv·√T), ∂z/∂atm_iv = -z/atm_iv.
        // Combined: ∂w/∂atm_iv = 2·atm_iv·T·f − atm_iv·T·z·f'·1
        //                       = atm_iv·T·(2·f − z·f')
        // ∂w/∂params_p = atm_iv²·T·∂f/∂params_p
        for (Size i = 1; i < N; ++i) {
            for (Size j = 0; j < n_k; ++j) {
                Real k = kGrid[j];
                Size base = ((i - 1) * n_k + j) * row;

                // Lower pillar contribution: gradient w.r.t. slice (i-1)
                {
                    const auto& s = slices_[i - 1];
                    Real T = T_[i - 1];
                    Real sigmaHat = s.atmIv * std::sqrt(T);
                    Real z = k / sigmaHat;
                    Real fv = shape_->f(z, s.params);
                    Real dfdz1 = shape_->dfdz(z, s.params);
                    auto dfdp = shape_->dfdParams(z, s.params);
                    Real datm = s.atmIv * T * (2.0 * fv - z * dfdz1);
                    // Sign: deficit = w_i − w_{i-1}, so wrt w_{i-1} → negate
                    out[base + 0] = -datm;
                    for (Size p = 0; p < n_shape; ++p) {
                        out[base + 1 + p] = -(s.atmIv * s.atmIv) * T * dfdp[p];
                    }
                }
                // Upper pillar contribution: gradient w.r.t. slice i
                {
                    const auto& s = slices_[i];
                    Real T = T_[i];
                    Real sigmaHat = s.atmIv * std::sqrt(T);
                    Real z = k / sigmaHat;
                    Real fv = shape_->f(z, s.params);
                    Real dfdz1 = shape_->dfdz(z, s.params);
                    auto dfdp = shape_->dfdParams(z, s.params);
                    Real datm = s.atmIv * T * (2.0 * fv - z * dfdz1);
                    out[base + grad_per_slice + 0] = datm;
                    for (Size p = 0; p < n_shape; ++p) {
                        out[base + grad_per_slice + 1 + p] =
                            (s.atmIv * s.atmIv) * T * dfdp[p];
                    }
                }
            }
        }
        return out;
    }

}  // namespace QuantLib
