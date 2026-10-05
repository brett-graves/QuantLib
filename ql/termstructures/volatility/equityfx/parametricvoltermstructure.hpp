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

/*! \file parametricvoltermstructure.hpp
    \brief Klassen-style parametric Black vol surface.

    Per-pillar slice carries an ATF implied vol and a vector of opaque
    "shape" parameters.  The smile is given by a subclassable
    ParametricVolShape object, which returns the dimensionless shape
    function

        f(z; p),   z := log(K/F) / (atm_iv · sqrt(T))   (Klassen normalisation)

    with f(0) = 1.  Implied vol and total variance follow as

        sigma^2(T, K) = atm_iv^2 * f(z; p)
        w (T, y)      = T * sigma^2 = T * atm_iv^2 * f(z; p),   y = log(K/F).

    Across maturities, total variance is interpolated linearly in T at
    fixed log-forward moneyness y; left-extrapolation shrinks linearly to
    w = 0 at t = 0, right-extrapolation holds the slope of the last
    interval (no parameter extrapolation).

    Calendar arb-freeness is the caller's responsibility — supply slices
    such that w(y, T) is non-decreasing in T at every y of interest.

    To plug in a custom shape, subclass ParametricVolShape in C++ or, via
    SWIG director, in Python and implement f(z, params) and dfdz(z, params).
*/

#ifndef quantlib_parametric_vol_term_structure_hpp
#define quantlib_parametric_vol_term_structure_hpp

#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/instruments/dividendschedule.hpp>
#include <ql/quote.hpp>
#include <ql/handle.hpp>
#include <ql/shared_ptr.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/utilities/null.hpp>
#include <vector>

namespace QuantLib {

    //! Abstract dimensionless smile shape (Klassen z-parametrization).
    /*! Implementations return f(z; params), the dimensionless smile
        relative to ATF, where z = log(K/F) / (atm_iv * sqrt(T)) is the
        normalized log-moneyness.  Total variance is then

            w(T, y) = T * atm_iv^2 * f(z; params)

        with f(0) = 1 by convention.

        Subclass in C++, or in Python via SWIG director, and override
        f() and dfdz().
    */
    class ParametricVolShape {
      public:
        virtual ~ParametricVolShape() = default;

        //! Dimensionless shape f(z; params).  Must satisfy f(0) = 1.
        virtual Real f(Real z, const std::vector<Real>& params) const = 0;

        //! Strike derivative d f / d z (in normalized space).
        virtual Real dfdz(Real z, const std::vector<Real>& params) const = 0;

        //! Second strike derivative d² f / d z² (normalized space).
        /*! Default implementation: central FD on `dfdz` with step `h`.
            Built-in shapes (S3, JW) override with closed forms; Python
            subclasses fall back to FD automatically.  The `h` knob is
            ignored by analytic overrides and exists only so the FD path
            remains tunable from the surface (butterflyDensity, Dupire). */
        virtual Real d2fdz2(Real z,
                            const std::vector<Real>& params,
                            Real h = 1e-4) const;

        //! Parameter Jacobian: vector of ∂f/∂params_i at fixed z.
        /*! Default implementation: central FD on `f`.  Built-in shapes
            (S3, JW) override with closed forms; Python subclasses fall
            back to FD automatically. */
        virtual std::vector<Real> dfdParams(
            Real z, const std::vector<Real>& params) const;

        //! f, df/dz and d²f/dz² (default FD step) at n points.
        /*! Default implementation: f(), dfdz() and d2fdz2() per point,
            so Python shapes keep their own overrides.  Built-in shapes
            override it with one pass that evaluates their shared
            intermediates once; the values equal the per-point calls
            exactly.  Not exposed to SWIG. */
        virtual void fSlice(const Real* z, Size n,
                            const std::vector<Real>& params,
                            Real* f, Real* dfdz, Real* d2fdz2) const;
    };

    //! S3 / SSVI shape: 3-parameter Klassen / Gatheral-Jacquier smile.
    /*! params = (s2, c2) with c2 >= 0 (s2 unconstrained but typically
        negative for equities).  Closed form:

            f(z) = 0.5 * (1 + s2*z)
                   + sqrt( 0.25 * (1 + s2*z)^2 + 0.5 * c2 * z^2 )

        f(0) = 1, f'(0) = s2, f''(0) = c2 - s2^2/2  (Klassen's "natural"
        dimensionless skew and curvature definitions match s2 and c2).
        Butterfly arb-free on the full (sigma_hat0, s2, c2) cube via the
        S3 closed-form bound (see Klassen 2017 §3).
    */
    class S3Shape : public ParametricVolShape {
      public:
        //! params = (s2, c2), c2 >= 0.
        Real f(Real z, const std::vector<Real>& params) const override;
        Real dfdz(Real z, const std::vector<Real>& params) const override;
        Real d2fdz2(Real z, const std::vector<Real>& params,
                    Real h = 1e-4) const override;
        std::vector<Real> dfdParams(
            Real z, const std::vector<Real>& params) const override;
        void fSlice(const Real* z, Size n, const std::vector<Real>& params,
                    Real* f, Real* dfdz, Real* d2fdz2) const override;
    };

    //! JW (Jump-Wing) shape: 5-parameter Klassen-style smile with
    //! independent left/right wing curvatures.
    /*! params = (s2, c_minus, c_plus) with c_minus, c_plus >= 0.
        The slice atmIv carries the ATF level so 4 shape params + atmIv
        = 5 free parameters per slice.

        Piecewise S3:

            f(z) = 0.5 * (1 + s2*z)
                   + sqrt( 0.25 * (1 + s2*z)^2 + 0.5 * c(z) * z^2 )

        with  c(z) = c_minus  if z < 0  else  c_plus.

        f(0) = 1, f'(0) = s2 (continuous from both sides; f''(0) is
        discontinuous when c_minus != c_plus, but bounded and harmless
        for fitting).

        Wing asymptotics (z -> ±inf):
            C_plus  = 0.5*s2 + sqrt(0.25*s2^2 + 0.5*c_plus)
            C_minus = sqrt(0.25*s2^2 + 0.5*c_minus) - 0.5*s2

        Reduces to S3 when c_minus = c_plus = c2.  Use for liquid
        underliers (ES/SPX/SPY/AAPL) where independent wing slopes are
        empirically required.
    */
    class JWShape : public ParametricVolShape {
      public:
        //! params = (s2, c_minus, c_plus), both c's >= 0.
        Real f(Real z, const std::vector<Real>& params) const override;
        Real dfdz(Real z, const std::vector<Real>& params) const override;
        Real d2fdz2(Real z, const std::vector<Real>& params,
                    Real h = 1e-4) const override;
        std::vector<Real> dfdParams(
            Real z, const std::vector<Real>& params) const override;
        void fSlice(const Real* z, Size n, const std::vector<Real>& params,
                    Real* f, Real* dfdz, Real* d2fdz2) const override;
    };

    //! K5 shape: tanh^2 blend between independent ATF curvature and
    //! per-side wing curvatures.
    /*! params = (s, c, c_minus, c_plus), all >= 0 except s (skew).
        Four shape params + atmIv = 5 free parameters per slice.

        Decouples ATF curvature `c` from wing slopes via a single
        tanh^2 blend with fixed width mu = 2.0 (centered on z = +/- 2sigma):

            f(z)        = 0.5 * (1 + s*z) + sqrt( R(z) )
            R(z)        = 0.25 * (1+sz)^2 + 0.5 * c_eff(z) * z^2
            c_eff(z)    = c + (c_wing(z) - c) * tanh^2(z / mu)
            c_wing(z)   = c_minus if z<0 else c_plus
            mu          = 2.0  (fixed)

        Properties:
          f(0)  = 1, f'(0) = s, f''(0) = c   (ATF identity + ATF curvature
                                              independent of c_wing)
          Wing asymptote (tanh^2 -> 1 as |z| -> inf):
            C_plus  = 0.5*s + sqrt(0.25*s^2 + 0.5*c_plus)
            C_minus = sqrt(0.25*s^2 + 0.5*c_minus) - 0.5*s
          R-positivity: c_eff = (1-T)*c + T*c_wing is a convex combination
          with non-negative weights, so c_eff >= 0 whenever
          c, c_wing >= 0 -- and hence R(z) >= 0 by construction.

        Use vs JW: JW couples ATF curvature and wing slope through the
        same per-side `c` parameter, so flat ATF + high wings is
        unreachable.  K5 separates the two with a smooth tanh^2 blend,
        which is exactly what symmetric / U-shaped smiles (TLT, XLF,
        bond/credit ETFs) need.

        Reduces to S3 when c = c_minus = c_plus.
    */
    class K5Shape : public ParametricVolShape {
      public:
        //! params = (s, c, c_minus, c_plus), c/c_minus/c_plus >= 0.
        Real f(Real z, const std::vector<Real>& params) const override;
        Real dfdz(Real z, const std::vector<Real>& params) const override;
        Real d2fdz2(Real z, const std::vector<Real>& params,
                    Real h = 1e-4) const override;
        std::vector<Real> dfdParams(
            Real z, const std::vector<Real>& params) const override;
        void fSlice(const Real* z, Size n, const std::vector<Real>& params,
                    Real* f, Real* dfdz, Real* d2fdz2) const override;
    };

    //! K7 shape: two-knot tanh-blended smile decoupling ATF / mid-wing /
    //! far-wing curvatures.
    /*! params = (s, c, c_mid_minus, c_mid_plus, c_far_minus, c_far_plus),
        all c's >= 0.  Six shape params + atmIv = 7 total per slice.

        Motivation: JW (3-param) and S3 (2-param) have a unique curvature
        maximum at ATF.  On long-dated SPX/SPY LEAPs the market wants HIGH
        curvature at moderate |z| and a PLATEAU (or reduced curvature) at
        far OTM.  K5's single tanh^2(z/2) blend is also dormant in the
        LEAP data range (small z due to long T).  K7 introduces a SECOND,
        tighter blend knot so the smile can ramp up between mu_1 = 0.5
        (inner-wing) and mu_2 = 2.0 (outer-wing) independently.

        Closed form:

            f(z)        = 0.5 * (1 + s*z) + sqrt( R(z) )
            R(z)        = 0.25 * (1+sz)^2 + 0.5 * c_eff(z) * z^2
            c_eff(z)    = c
                        + (c_mid(z) - c) * tanh^2(z / mu_1)
                        + (c_far(z) - c_mid(z)) * tanh^2(z / mu_2)
            c_mid(z)    = c_mid_minus if z<0 else c_mid_plus
            c_far(z)    = c_far_minus if z<0 else c_far_plus
            mu_1        = 0.5,  mu_2 = 2.0  (fixed)

        Properties:
          f(0)  = 1, f'(0) = s, f''(0) = c   (ATF identity + ATF curvature)
          Wing asymptote: f(z) ~ |z| * C_pm with
            C_plus  = 0.5*s + sqrt(0.25*s^2 + 0.5*c_far_plus)
            C_minus = sqrt(0.25*s^2 + 0.5*c_far_minus) - 0.5*s
          R-positivity: c_eff is a convex combination of (c, c_mid, c_far)
          with non-negative weights (1-T_1, T_1-T_2, T_2), so c_eff >= 0
          when all three are >= 0 -- and hence R(z) >= 0 by construction.

          c_far governs the Lee bound; c_mid is "transient" mid-wing
          curvature that doesn't affect the asymptote.  This decouples
          mid-wing acceleration from far-wing slope, exactly the
          flexibility long-dated SPX/SPY smiles need.

        Reduces to S3 when c = c_mid_pm = c_far_pm.
        Approaches K5 (single blend at mu=2) when c_mid_pm = c.
    */
    class K7Shape : public ParametricVolShape {
      public:
        //! params = (s, c, c_mid_minus, c_mid_plus, c_far_minus, c_far_plus),
        //! all c's >= 0.
        Real f(Real z, const std::vector<Real>& params) const override;
        Real dfdz(Real z, const std::vector<Real>& params) const override;
        Real d2fdz2(Real z, const std::vector<Real>& params,
                    Real h = 1e-4) const override;
        std::vector<Real> dfdParams(
            Real z, const std::vector<Real>& params) const override;
    };

    //! Wing ramp sharpness guard for SplShape (see the class docs).
    const Real SPL_MAX_ATM_LEAK = 1e-2;
    //! Finite-difference-probe floor for SplShape belly control points.
    const Real SPL_COEF_FLOOR = 1e-8;
    //! SplShape belly B-spline degree. Quintic => C4 across interior knots.
    const Size SPL_BSPLINE_DEGREE = 5;
    //! Power coefficients per interval: a quintic needs 6 (u^0 .. u^5).
    const Size SPL_N_POW = SPL_BSPLINE_DEGREE + 1;

    //! SPL: quintic-B-spline belly + Lee-boxed softplus wings.
    /*! Unlike S3/JW/K5/K7 this shape is not a fixed closed form: the belly is a
        spline over user-configured knots, so the parameter count depends on the
        knot grid supplied at construction.

            f(z)    = base(z) + F_L(z) + F_R(z)
            base(z) = sum_j c_j B_j(z)      (clamped quintic B-spline)
            F_R(z)  = sR * softplus_tau(z - xR)
            F_L(z)  = sL * softplus_tau(xL - z)

        Why a B-spline rather than a spline INTERPOLATING node values:

        - POSITIVITY.  B-spline basis functions are non-negative and partition
          unity, so all c_j > 0 implies base(z) > 0 EVERYWHERE, not merely at the
          knots.  Since the wings are non-negative too, f > 0 everywhere.  An
          interpolating spline is positive only AT its nodes and can dip negative
          between them -- a negative variance ratio, i.e. negative total
          variance, which no surface can price.  chloride log-transforms these
          channels, so c_j > 0 comes free from the existing parametrization.
        - NO OSCILLATION.  The curve lies in the convex hull of its control
          points, so it cannot overshoot chasing node values.
        - SMOOTHNESS.  Degree 5, so simple interior knots give C4.  A cubic
          basis gives only C2, leaving an O(1) jump in the THIRD derivative at
          every knot.  Dupire differentiates total variance twice in strike, so
          those jumps were measured at 360-441x the between-knot curvature
          background -- ridges in the local-vol surface, which gamma and vanna
          then differentiate again.  C2 is enough for local vol to EXIST; it is
          not enough for it to be smooth.  Raising the degree costs no
          oscillation here because the curve still lies in the convex hull of
          its control points.

        The first three and last three control points are tied so
        base' = base'' = 0 at the outer knots; beyond them the belly is constant,
        so that join is C2 and the wings own all asymptotic slope.  As z -> +/-inf
        the wing slopes are sR / -sL, so the total-variance wing slope is
        beta± = atm_iv*sqrt(T)*s±, and Lee's bound beta± <= 2 becomes the box
        s± in [0, 2/(atm_iv*sqrt(T))] -- enforced by the calibrator, not here.

        `tau` (wing ramp sharpness) is CONSTRUCTION CONFIG, not a fitted channel,
        for the same reason K5Shape hard-codes its blend width: fitting it makes
        the wings degenerate with the belly near ATM.  A small tau smears the
        softplus to the money, where it swamps the f(0)=1 pin -- fatal at short
        maturities, where the Lee bound permits very large wing slopes.  The
        constructor rejects any (tau, switch) pair leaking more than
        SPL_MAX_ATM_LEAK per unit of wing slope.

        The ATM level is set by NORMALISATION rather than by pinning a control
        point: base(z) = (1 - wings(0)) * S(z)/S(0) with S(z) = sum_j c_j B_j(z),
        so f(0) = 1 identically for any positive control points.  c_0 is fixed at
        1 because base is invariant under a common rescaling of the c_j.

        params = (c_1, ..., c_{m}, sL, sR).

        The belly is LINEAR in the control points, so d base / d c_j is just
        B_j(z) -- dfdParams is analytic and needs no finite differences.
    */
    class SplShape : public ParametricVolShape {
      public:
        /*! \param knots        strictly increasing belly breakpoints in z
            \param switchLeft   left wing onset; Null => knots.front()
            \param switchRight  right wing onset; Null => knots.back()
            \param tau          wing ramp sharpness (config, not fitted)
        */
        explicit SplShape(std::vector<Real> knots,
                          Real switchLeft = Null<Real>(),
                          Real switchRight = Null<Real>(),
                          Real tau = 3.0);

        Real f(Real z, const std::vector<Real>& params) const override;
        Real dfdz(Real z, const std::vector<Real>& params) const override;
        Real d2fdz2(Real z, const std::vector<Real>& params,
                    Real h = 1e-4) const override;
        std::vector<Real> dfdParams(
            Real z, const std::vector<Real>& params) const override;

        //! Number of shape parameters implied by the knot grid.
        Size nParams() const { return nNode_ + 2; }
        const std::vector<Real>& knots() const { return knots_; }
        Real tau() const { return tau_; }

      private:
        Size intervalAt(Real z) const;
        //! Reduced (tied) basis functions / derivatives at z.
        std::vector<Real> phiAt(Real z, Size order) const;
        void ensureCoef(const std::vector<Real>& params) const;
        Real bellyEval(Real z, Size order) const;

        std::vector<Real> knots_;
        Size nNode_ = 0, nFree_ = 0;
        Real xLeft_ = 0.0, xRight_ = 0.0, zl_ = 0.0, zr_ = 0.0, tau_ = 3.0;
        std::vector<Real> mids_;
        //! cmap_[interval][power][control point]: constant for a given knot grid.
        std::vector<std::vector<std::vector<Real> > > cmap_;
        std::vector<Real> phi0_;   //!< reduced basis at z = 0 (for the ATM pin)

        // 1-entry cache keyed on the params vector (rebuilt once per fit pass).
        mutable std::vector<Real> cacheKey_;
        mutable std::vector<std::vector<Real> > poly_;  // per interval, SPL_N_POW each
        mutable std::vector<Real> coef_;
        mutable Real sLc_ = 0.0, sRc_ = 0.0, scale_ = 1.0, wings0_ = 0.0;
        mutable bool cacheValid_ = false;
    };

    //! Per-pillar slice spec for ParametricVolTermStructure.
    struct ParametricVolSlice {
        Real atmIv;                  //!< ATF implied vol at this maturity, > 0
        std::vector<Real> params;    //!< shape params, opaque to the surface
    };

    //! Black vol surface from a subclassable Klassen-style smile shape.
    class ParametricVolTermStructure : public BlackVolatilityTermStructure {
      public:
        //! Construct (continuous dividend yield only).
        ParametricVolTermStructure(
            const Date& referenceDate,
            const std::vector<Date>& dates,
            const std::vector<ParametricVolSlice>& slices,
            ext::shared_ptr<ParametricVolShape> shape,
            Handle<Quote> spot,
            Handle<YieldTermStructure> riskFreeRate,
            Handle<YieldTermStructure> dividendYield,
            const DayCounter& dc = Actual365Fixed());

        //! Construct with discrete dividends.
        ParametricVolTermStructure(
            const Date& referenceDate,
            const std::vector<Date>& dates,
            const std::vector<ParametricVolSlice>& slices,
            ext::shared_ptr<ParametricVolShape> shape,
            Handle<Quote> spot,
            Handle<YieldTermStructure> riskFreeRate,
            Handle<YieldTermStructure> dividendYield,
            DividendSchedule dividends,
            const DayCounter& dc = Actual365Fixed());

        //! \name TermStructure interface
        //@{
        Date maxDate() const override { return Date::maxDate(); }
        //@}

        //! \name VolatilityTermStructure interface
        //@{
        Real minStrike() const override { return QL_MIN_REAL; }
        Real maxStrike() const override { return QL_MAX_REAL; }
        //@}

        //! \name Inspectors
        //@{
        Size numSlices() const { return slices_.size(); }
        const std::vector<Time>& maturities() const { return T_; }
        const std::vector<ParametricVolSlice>& slices() const { return slices_; }
        const ParametricVolSlice& slice(Size i) const { return slices_.at(i); }
        const ext::shared_ptr<ParametricVolShape>& shape() const { return shape_; }
        //@}

        //! \name Direct access to the shape (bypasses BlackVol interface)
        //@{
        //! Normalized strike z at pillar i for log-moneyness k.
        Real z(Size sliceIdx, Real k) const;
        //! Total variance at pillar slice i for log-moneyness k.
        Real totalVariance(Size sliceIdx, Real k) const;
        //! Total variance at arbitrary (k, t), interpolated in T.
        Real totalVariance(Real k, Time t) const;
        //! d w / d k at pillar slice i.
        Real totalVarianceStrikeDerivative(Size sliceIdx, Real k) const;
        //! d w / d k at arbitrary (k, t), interpolated in T.
        Real totalVarianceStrikeDerivative(Real k, Time t) const;
        //! d² w / d k² at pillar slice i.
        Real totalVarianceStrikeSecondDerivative(Size sliceIdx, Real k) const;
        //! d² w / d k² at arbitrary (k, t), interpolated in T.
        Real totalVarianceStrikeSecondDerivative(Real k, Time t) const;
        //! ∂w/∂T at fixed log-moneyness k.  Right-continuous on pillars.
        /*! Linear-in-w time interpolation gives a piecewise-constant
            slope per pillar interval.  Left-extrapolation slope is
            w(slice 0, k) / T_0; right-extrapolation reuses the slope
            of the last interior interval.  Throws on a single-pillar
            surface (Dupire is degenerate). */
        Real totalVarianceTimeDerivative(Real k, Time t) const;
        //! Forward F(t) used internally for strike <-> k.
        Real forward(Time t) const;
        //@}

        //! \name Analytic Dupire local volatility
        //@{
        /*! σ²_loc(K, T) at log-moneyness k = log(K / F(T)) and maturity
            t > 0.  Uses the standard Gatheral form

              σ²_loc = ∂w/∂T / [ 1 − k/w·∂w/∂k
                                + ¼(−¼ − 1/w + k²/w²)·(∂w/∂k)²
                                + ½·∂²w/∂k² ]

            Throws if the denominator is non-positive (a calendar or
            butterfly arb violation has corrupted the Dupire identity)
            or the surface has fewer than two pillars. */
        Real localVariance(Real k, Time t) const;
        //! σ_loc(K, T) = √localVariance(k, t).
        Real localVol(Real k, Time t) const;
        //! localVariance(k[i], t) for i < n, written to out[i].
        /*! The per-time work (pillar weights, the pillar interval whose
            slope is ∂w/∂T, the slice scales) is done once for all points,
            and the total variance of each bracketing pillar is shared
            between w and ∂w/∂T.  Values are identical to localVariance().
            Where localVariance() would throw on the point (non-positive
            total variance or Dupire denominator) or localVol() would
            reject it (negative local variance), out[i] is Null<Real>()
            instead; the return value is the number of such points.  A
            single-pillar surface has no ∂w/∂T, so every point is Null.
            k and out may be the same array. */
        Size localVarianceSlice(Time t, const Real* k, Size n,
                                Real* out) const;
        //@}

        //! \name Mutators
        //@{
        //! Replace (atm_iv, params) at pillar i and notify observers.
        void setSlice(Size i, Real atmIv, const std::vector<Real>& params);
        //! Replace all slice (atm_iv, params), keeping pillar dates.
        void setSlices(const std::vector<ParametricVolSlice>& slices);
        //@}

        //! \name Batch evaluation (for sequential / global LM fitters)
        //@{
        /*! Black vols at a list of (sliceIdx, strike) pairs.  Skips the
            time interpolation — evaluates each observation directly at
            its pillar.  Useful when the fitter walks a single slice. */
        std::vector<Real> batchBlackVol(
            const std::vector<Size>& sliceIndices,
            const std::vector<Real>& strikes) const;

        /*! Black vols at a list of (maturity, strike) pairs with time
            interpolation.  Mirrors EssviVolatilityTermStructure's
            equivalent batch API. */
        std::vector<Real> batchBlackVolAtTimes(
            const std::vector<Real>& times,
            const std::vector<Real>& strikes) const;

        /*! Per-observation gradient ∂σ/∂(atm_iv, params...) at each
            (sliceIdx, K).  Returns flat row-major array of size
            n_obs · (1 + n_shape) where n_shape = slice(0).params.size().

            Layout per row: [∂σ/∂atm_iv, ∂σ/∂params[0], ..., ∂σ/∂params[n_shape-1]]

            Gradients are local to the obs's slice (no cross-pillar
            coupling).  Caller scatters into the global Jacobian using
            sliceIndices. */
        std::vector<Real> batchImpliedVolGradient(
            const std::vector<Size>& sliceIndices,
            const std::vector<Real>& strikes) const;

        //! Number of shape params per slice (all slices share the same count).
        Size nShapeParams() const {
            return slices_.empty() ? 0 : slices_.front().params.size();
        }
        //@}

        //! \name Arbitrage checks (Klassen 2017 §3)
        //@{
        /*! Implied density g(z) at pillar i for normalized strike z.
            Computed via the Klassen formula

              g(z) = (1 − z·f'/(2f))² − ¼·(f')²/f
                     − (sigma_hat0²/16)·(f')²
                     + ½·f''

            with f, f' from the shape callable and f'' from a central
            FD on f' (so this works for any shape, including Python
            subclasses).  Butterfly arb-free <=> g(z) >= 0 at every z. */
        Real butterflyDensity(Size sliceIdx, Real z, Real h = 1e-4) const;

        /*! Maximum butterfly-arb violation magnitude at pillar i,
            scanned over the supplied z grid.  Returns
              max( 0, -min_g )
            where min_g is the minimum g(z) on the grid.  Zero <=>
            no butterfly violation on the grid. */
        Real butterflyArbViolation(Size sliceIdx,
                                   const std::vector<Real>& zGrid,
                                   Real h = 1e-4) const;

        /*! Maximum calendar-arb violation magnitude across consecutive
            pillar pairs, scanned at the supplied log-moneyness grid.
            Returns
              max_i max_k max(0, w(T_i, k) - w(T_{i+1}, k))
            Zero <=> no calendar violation at any (k, pillar-pair) on
            the grid. */
        Real calendarArbViolation(const std::vector<Real>& kGrid) const;
        //@}

        //! \name Constraint-grid evaluators (for SLSQP / trust-constr)
        //@{
        /*! Butterfly density g(z) at every (pillar i, z in zGrid).

            Returns flat row-major array of size N · n_z.  Row major:
            [i=0,z=0], [i=0,z=1], ..., [i=0,z=n_z-1], [i=1,z=0], ...

            Use as the "value" output of a scipy-style inequality
            constraint `g(z) >= 0`. */
        std::vector<Real> butterflyDensityGrid(
            const std::vector<Real>& zGrid, Real h = 1e-4) const;

        /*! Jacobian of butterflyDensityGrid w.r.t. each slice's own
            (atm_iv, params) — sparse per slice, no cross-pillar
            coupling.  Returns flat array of size N · n_z · (1 + n_shape).

            Element layout: row-major (slice, z, param) — for each row
            (i, j), the (1 + n_shape) gradient w.r.t. slice i's
            (atm_iv_i, params_i[0..n_shape-1]).

            Caller scatters into the full (N · n_z, N · (1 + n_shape))
            Jacobian using slice_i for each row. */
        std::vector<Real> butterflyDensityGridGradient(
            const std::vector<Real>& zGrid, Real h = 1e-4) const;

        /*! Calendar deficits w_{i}(k) - w_{i-1}(k) at every
            (pillar pair, k in kGrid) for i = 1 .. N-1.

            Returns flat row-major of size (N-1) · n_k.  Non-negative
            values mean the surface is calendar-arb-free at that (k,
            pair).  Use directly as a scipy inequality constraint
            `deficit >= 0`. */
        std::vector<Real> calendarDeficitGrid(
            const std::vector<Real>& kGrid) const;

        /*! Jacobian of calendarDeficitGrid w.r.t. all per-slice params.

            For pair (i-1, i) row j (k = kGrid[j]):
              ∂(w_i - w_{i-1})/∂params_i      → +∂w_i/∂params_i
              ∂(w_i - w_{i-1})/∂params_{i-1}  → -∂w_{i-1}/∂params_{i-1}
            All other entries zero.

            Returns flat array of size (N-1) · n_k · 2 · (1 + n_shape).
            Row layout for each (pair_idx, k):
              [grad_w_lower (1+n_shape), grad_w_upper (1+n_shape)]
            Caller scatters using the pair indices (i-1, i). */
        std::vector<Real> calendarDeficitGridGradient(
            const std::vector<Real>& kGrid) const;
        //@}

      protected:
        Volatility blackVolImpl(Time t, Real strike) const override;

      private:
        struct TimeWeights {
            Size sLo, sHi;
            Real wLo, wHi;
            bool leftExtrap;
        };
        TimeWeights timeWeights(Time t) const;

        std::vector<Time>                  T_;
        std::vector<ParametricVolSlice>    slices_;
        ext::shared_ptr<ParametricVolShape> shape_;
        Handle<Quote>                      spot_;
        Handle<YieldTermStructure>         riskFreeRate_;
        Handle<YieldTermStructure>         dividendYield_;
        DividendSchedule                   dividends_;
    };

} // namespace QuantLib

#endif
