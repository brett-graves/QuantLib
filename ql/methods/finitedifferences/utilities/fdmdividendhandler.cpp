/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008, 2009 Ralph Schreyer
 Copyright (C) 2008 Klaus Spanderen

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

#include <ql/time/daycounter.hpp>
#include <ql/math/interpolations/cubicinterpolation.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdividendhandler.hpp>

namespace QuantLib {


    FdmDividendHandler::FdmDividendHandler(
        const DividendSchedule& schedule,
        const ext::shared_ptr<FdmMesher>& mesher,
        const Date& referenceDate,
        const DayCounter& dayCounter,
        Size equityDirection)
    : x_(mesher->layout()->dim()[equityDirection]),
      mesher_(mesher),
      equityDirection_(equityDirection) {

        dividends_.reserve(schedule.size());
        dividendCashflows_.reserve(schedule.size());
        dividendDates_.reserve(schedule.size());
        dividendTimes_.reserve(schedule.size());
        for (const auto& iter : schedule) {
            // FixedDividend::amount() returns the constant payout, used as
            // an informational accessor + cached for callers that want
            // pre-computed cash amounts.  FractionalDividend::amount()
            // throws unless a nominal was preset; in that case we keep 0
            // here -- the per-grid-node drop is computed in applyTo via
            // amount(underlying) so it scales with simulated spot.
            const auto& fd = ext::dynamic_pointer_cast<FractionalDividend>(iter);
            if (fd && fd->nominal() == Null<Real>())
                dividends_.push_back(0.0);
            else
                dividends_.push_back(iter->amount());
            dividendCashflows_.push_back(iter);
            dividendDates_.push_back(iter->date());
            dividendTimes_.push_back(dayCounter.yearFraction(referenceDate, iter->date()));
        }

         Array tmp = mesher_->locations(equityDirection);
         Size spacing = mesher_->layout()->spacing()[equityDirection];
         for (Size i = 0; i < x_.size(); ++i) {
             x_[i] = std::exp(tmp[i*spacing]);
         }
    }

    const std::vector<Time>& FdmDividendHandler::dividendTimes() const {
        return dividendTimes_;
    }
         
    const std::vector<Date>& FdmDividendHandler::dividendDates() const {
        return dividendDates_;
    }

    const std::vector<Real>& FdmDividendHandler::dividends() const {
        return dividends_;
    }

    void FdmDividendHandler::applyTo(Array& a, Time t) const {
        Array aCopy(a);

        auto iter = std::find(dividendTimes_.begin(), dividendTimes_.end(), t);

        if (iter != dividendTimes_.end()) {
            // Dispatch to Dividend::amount(underlying) so FractionalDividend
            // scales with the simulated spot at the ex-date (rate * spot)
            // instead of being frozen at a single nominal at construction.
            // FixedDividend ignores the argument and returns its constant
            // amount, preserving legacy behaviour for cash divs.
            const auto& div = dividendCashflows_[iter - dividendTimes_.begin()];

            // The jump V(S) <- V(S - D) reads the value between nodes.
            // Linear interpolation of a convex value function always reads
            // high, by O(h^2) per ex-date, and the error accumulates over
            // every dividend in the option's life (chloride #593: +$0.15 on
            // a 2y SPY call at 150 nodes).  A monotone cubic spline in
            // ln S -- the interpolant Fdm1DimSolver reads the result with --
            // removes that term; the Hyman filter keeps it from overshooting
            // near a payoff kink when the ex-date is close to expiry.
            Array lnX(x_.size());
            Array lnTarget(x_.size());
            for (Size k=0; k<x_.size(); ++k) {
                lnX[k] = std::log(x_[k]);
                lnTarget[k] = std::log(
                    std::max(x_[0], x_[k] - div->amount(x_[k])));
            }

            if (mesher_->layout()->dim().size() == 1) {
                MonotonicCubicNaturalSpline interp(
                    lnX.begin(), lnX.end(), aCopy.begin());
                for (Size k=0; k<x_.size(); ++k)
                    a[k] = interp(lnTarget[k], true);
            }
            else {
                Array tmp(x_.size());
                Size xSpacing = mesher_->layout()->spacing()[equityDirection_];

                for (Size i=0; i<mesher_->layout()->dim().size(); ++i) {
                    if (i!=equityDirection_) {
                        Size ySpacing = mesher_->layout()->spacing()[i];
                        for (Size j=0; j<mesher_->layout()->dim()[i]; ++j) {
                            for (Size k=0; k<x_.size(); ++k) {
                                Size index = j*ySpacing + k*xSpacing;
                                tmp[k] = aCopy[index];
                            }
                            MonotonicCubicNaturalSpline interp(
                                lnX.begin(), lnX.end(), tmp.begin());
                            for (Size k=0; k<x_.size(); ++k) {
                                Size index = j*ySpacing + k*xSpacing;
                                a[index] = interp(lnTarget[k], true);
                            }
                        }
                    }
                }
            }
        }
    }
}
