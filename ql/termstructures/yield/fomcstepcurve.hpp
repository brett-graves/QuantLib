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

/*! \file fomcstepcurve.hpp
    \brief Overnight-rate curve fitted by least squares with FOMC-step policy moves
*/

#ifndef quantlib_fomc_step_curve_hpp
#define quantlib_fomc_step_curve_hpp

#include <ql/math/array.hpp>
#include <ql/math/matrix.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/time/calendar.hpp>
#include <string>
#include <utility>
#include <vector>

namespace QuantLib {

    //! Log-linear discount curve whose node discounts are set in place
    /*! Between nodes the instantaneous forward is flat. The fit below places
        a node on every day the overnight rate can change, so a step path is
        reproduced exactly. Setting the discounts notifies observers (rate
        helpers, instruments) as a relink would.
    */
    class FomcStepNodeCurve : public InterpolatedDiscountCurve<LogLinear> {
      public:
        FomcStepNodeCurve(const std::vector<Date>& dates, const DayCounter& dayCounter);
        void setDiscounts(const std::vector<DiscountFactor>& discounts);
    };

    //! Shape and regularisation of the FOMC-step fit
    /*! Turn priors (mean, sd in bp, month-, quarter- and year-end) come from
        SOFR fixings 2022-06..2026-09 with FOMC windows excluded; the turn
        shape weights business days 0, +1, +2 after a month's last.
    */
    struct FomcStepCurveConfig {
        Real horizonYears = 3.0;
        std::vector<Real> splineKnotsYears = {4, 5, 6, 7, 8.5, 10, 12, 15, 20, 25, 30, 40, 50};
        Real pathSigmaBp = 25.0;
        std::vector<Real> turnPriorMeanBp = {2.5, 4.5, 5.0};
        std::vector<Real> turnPriorSdBp = {3.0, 4.0, 6.0};
        std::vector<Real> turnShape = {1.0, 1.0, 0.5};
        //! inside its band a quote pulls toward mid with sd bandInnerK half-widths
        Real bandInnerK = 4.0;
        //! outside its band every bandWallBp of error costs one sd
        Real bandWallBp = 0.1;
        //! funding spread: sd of month-on-month change and of second difference
        Real spreadStepBp = 1.0;
        Real spreadCurveBp = 0.5;
        Real curvatureLambda = 1e-2;
        Integer curveYears = 52;
        Integer tailNodeDays = 30;
        Size maxOuterSteps = 8;
        Real nonlinearTolBp = 1e-3;
    };

    //! Overnight rates linear in the parameters: r = A x / 1e4 on business days
    /*! The SOFR overnight rate on business day b is

            policy(b) + s(b) + turn(b)       before the switch date
            S(t_b)                           from it,

        where policy(b) = L + sum_k D_k 1[b >= e_k] steps on each FOMC
        effective date e_k (the business day after the decision) inside the
        horizon, s is the funding spread (piecewise linear on knots at the
        reference date, the first of each month and the switch date), turns are
        month/quarter/year-end bumps shared by kind, and S is a clamped cubic
        B-spline in the forward tied to the last step level at the switch, the
        last meeting's effective date. With an EFFR leg the policy path is
        EFFR and today's spread is free; without one the spread is measured
        from today (first knot fixed at zero).

        Parameters, in bp: level, one move per meeting, the spread knots, the
        three turn sizes, then the spline coefficients after the first.
    */
    class FomcStepParametrisation {
      public:
        FomcStepParametrisation(const Date& referenceDate,
                                std::vector<Date> meetingDecisions,
                                Calendar calendar,
                                FomcStepCurveConfig config = {},
                                bool withEffr = false);

        Size size() const { return names_.size(); }
        const std::vector<std::string>& names() const { return names_; }
        const Date& referenceDate() const { return reference_; }
        const Date& switchDate() const { return switch_; }
        const std::vector<Date>& meetingDecisions() const { return decisions_; }
        const std::vector<Date>& meetingEffectiveDates() const { return effective_; }
        const std::vector<Date>& businessDays() const { return bdays_; }
        const std::vector<Date>& nodeDates() const { return nodeDates_; }
        const std::vector<Date>& spreadKnots() const { return spreadKnots_; }
        const FomcStepCurveConfig& config() const { return config_; }
        bool withEffr() const { return withEffr_; }

        Size levelIndex() const { return 0; }
        std::pair<Size, Size> movesRange() const { return moves_; }
        std::pair<Size, Size> spreadRange() const { return spread_; }
        std::pair<Size, Size> turnsRange() const { return turns_; }
        std::pair<Size, Size> splineRange() const { return spline_; }

        //! overnight rate (decimal) per business day; effr: the policy path
        std::vector<Real> rates(const Array& x, bool effr = false) const;
        //! discount factors on the node dates
        std::vector<DiscountFactor> nodeDiscounts(const Array& x, bool effr = false) const;
        //! d log DF(node) / dx for the SOFR curve, per bp of each parameter
        Matrix dLogDiscountDx(const Array& x) const;
        //! penalties(x) = P x + p0; every regulariser is linear
        const Matrix& penaltyMatrix() const { return penalty_; }
        const Array& penaltyOffset() const { return penaltyOffset_; }
        //! flat curve at overnightBp, turns at their priors
        Array initialGuess(Real overnightBp) const;

      private:
        typedef std::vector<std::pair<Size, Real> > Row;
        std::vector<Real> rowRates(const std::vector<Row>& rows, const Array& x) const;
        void buildPenalties(const Matrix& curvatureFactor, Size nSpline);

        Date reference_, switch_;
        Calendar calendar_;
        FomcStepCurveConfig config_;
        bool withEffr_;
        std::vector<Date> decisions_, effective_, days_, bdays_, nodeDates_, spreadKnots_;
        std::vector<Size> calToB_, nodeIndex_;
        std::vector<std::string> names_;
        std::pair<Size, Size> moves_, spread_, turns_, spline_;
        std::vector<Row> sofrRows_, effrRows_;
        Matrix penalty_;
        Array penaltyOffset_;
    };

    //! FOMC-step curve fitted to rate helpers by band-weighted least squares
    /*! Each target is a rate helper bound to the SOFR curve or, for an EFFR
        leg, to the policy curve; its error is (implied - quoted) / bpValue in
        bp of rate. A quote is scored against its uncertainty band (half a tick
        for a settle, half the spread for a live quote): a weak pull to mid
        inside, a steep wall outside. Synthetic targets sit out the solve and
        are then quoted at the fitted value (their quote must be a SimpleQuote),
        leaving the curve unchanged and adding risk knobs.

        Solver: Gauss-Newton on the quote errors linearised at the current
        point (the penalties are linear), the convex piecewise-quadratic band
        problem solved by an active-set loop (Levenberg-Marquardt if the set
        cycles), one re-pricing per step, the quote Jacobian re-evaluated only
        when the re-pricing departs from the linear model.

        Risk comes from the band-weighted quadratic model at the solution:
        covariance (J'J)^+, dx/dq per bp of each target's rate and
        d log DF(node) / dq, the curve's linear response to every mark.
    */
    class FomcStepFit {
      public:
        FomcStepFit(ext::shared_ptr<FomcStepParametrisation> parametrisation,
                    std::vector<ext::shared_ptr<RateHelper> > helpers,
                    std::vector<std::string> names,
                    std::vector<Real> bpValues,
                    std::vector<Real> halfWidthsBp,
                    std::vector<bool> onEffrCurve,
                    std::vector<bool> synthetic,
                    const DayCounter& dayCounter);

        //! solve from x0; loss "band" or "quadratic"
        void fit(const Array& x0,
                 const std::string& loss = "band",
                 bool holdSynthetic = false,
                 bool reuseJacobian = false);
        //! show the curves at parameters x (no refit)
        void show(const Array& x);

        const ext::shared_ptr<FomcStepParametrisation>& parametrisation() const { return param_; }
        ext::shared_ptr<YieldTermStructure> sofrCurve() const { return sofr_; }
        ext::shared_ptr<YieldTermStructure> effrCurve() const { return effr_; }
        const std::vector<std::string>& targetNames() const { return names_; }

        const Array& parameters() const { return x_; }
        const Array& shownParameters() const { return shown_; }
        const Array& residualsBp() const { return residualBp_; }
        const Matrix& covariance() const { return covariance_; }
        const Matrix& dxdq() const { return dxdq_; }
        const Matrix& dLogDiscountDq() const { return dLogDfDq_; }
        const Matrix& quoteJacobian() const { return jacobian_; }
        std::vector<std::string> outsideBand() const;
        Real cost() const { return cost_; }
        Size outerSteps() const { return outerSteps_; }

      private:
        Array quoteErrorsBp(const Array& x);
        Matrix jacobianAt(const Array& x, const Array& e);
        Array solveLinear(const Array& x,
                          const Array& e,
                          const Matrix& Je,
                          const Array& hw,
                          bool band) const;

        ext::shared_ptr<FomcStepParametrisation> param_;
        std::vector<ext::shared_ptr<RateHelper> > helpers_;
        std::vector<std::string> names_;
        std::vector<Real> bpValues_, halfWidthsBp_;
        std::vector<bool> onEffr_, synthetic_;
        std::vector<ext::shared_ptr<SimpleQuote> > syntheticQuotes_;
        ext::shared_ptr<FomcStepNodeCurve> sofr_, effr_;
        Array x_, shown_, residualBp_;
        Matrix covariance_, dxdq_, dLogDfDq_, jacobian_;
        Real cost_ = 0.0;
        Size outerSteps_ = 0;
    };

}

#endif
