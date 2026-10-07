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
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdividendhandler.hpp>

#include <algorithm>
#include <cmath>

namespace QuantLib {

    namespace detail {

        FdmDividendJump::FdmDividendJump(const std::vector<Real>& s,
                                         const Dividend& div)
        : first_(s.size()), lo_(s.size()), w_(s.size()) {
            const Size n = s.size();
            QL_REQUIRE(n >= 4, "dividend jump needs at least 4 grid nodes");
            std::vector<Real> x(n);
            for (Size k=0; k<n; ++k)
                x[k] = std::log(s[k]);
            for (Size k=0; k<n; ++k) {
                // Dividend::amount(underlying): FractionalDividend scales
                // with the spot at the ex-date (rate * s), FixedDividend
                // returns its constant amount.
                const Real t =
                    std::log(std::max(s[0], s[k] - div.amount(s[k])));
                // bracket x[i] <= t <= x[i+1]
                Size i = std::upper_bound(x.begin(), x.end(), t) - x.begin();
                i = std::min(i == 0 ? Size(0) : i - 1, n - 2);
                lo_[k] = i;
                const Size f = std::min(i == 0 ? Size(0) : i - 1, n - 4);
                first_[k] = f;
                for (Size m=0; m<4; ++m) {
                    Real w = 1.0;
                    for (Size l=0; l<4; ++l)
                        if (l != m)
                            w *= (t - x[f+l]) / (x[f+m] - x[f+l]);
                    w_[k][m] = w;
                }
            }
        }

        void FdmDividendJump::apply(const Array& in, Array& out) const {
            for (Size k=0; k<w_.size(); ++k) {
                const Size f = first_[k], i = lo_[k];
                const std::array<Real, 4>& w = w_[k];
                const Real v = w[0]*in[f] + w[1]*in[f+1]
                             + w[2]*in[f+2] + w[3]*in[f+3];
                const Real a = in[i], b = in[i+1];
                out[k] = std::min(std::max(v, std::min(a, b)),
                                  std::max(a, b));
            }
        }

        void FdmDividendJump::apply(const Matrix& in, Matrix& out) const {
            const Size cols = in.columns();
            for (Size k=0; k<w_.size(); ++k) {
                const Size f = first_[k], i = lo_[k];
                const std::array<Real, 4>& w = w_[k];
                const Real* y0 = in.row_begin(f);
                const Real* y1 = in.row_begin(f+1);
                const Real* y2 = in.row_begin(f+2);
                const Real* y3 = in.row_begin(f+3);
                const Real* ya = in.row_begin(i);
                const Real* yb = in.row_begin(i+1);
                Real* o = out.row_begin(k);
                for (Size j=0; j<cols; ++j) {
                    const Real v = w[0]*y0[j] + w[1]*y1[j]
                                 + w[2]*y2[j] + w[3]*y3[j];
                    o[j] = std::min(std::max(v, std::min(ya[j], yb[j])),
                                    std::max(ya[j], yb[j]));
                }
            }
        }

    }


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

         const std::vector<Real> s(x_.begin(), x_.end());
         jumps_.reserve(dividendCashflows_.size());
         for (const auto& d : dividendCashflows_)
             jumps_.emplace_back(s, *d);
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
            // See detail::FdmDividendJump (chloride #593).
            const detail::FdmDividendJump& jump =
                jumps_[iter - dividendTimes_.begin()];

            if (mesher_->layout()->dim().size() == 1) {
                jump.apply(aCopy, a);
            }
            else {
                Array tmp(x_.size()), jumped(x_.size());
                Size xSpacing = mesher_->layout()->spacing()[equityDirection_];

                for (Size i=0; i<mesher_->layout()->dim().size(); ++i) {
                    if (i!=equityDirection_) {
                        Size ySpacing = mesher_->layout()->spacing()[i];
                        for (Size j=0; j<mesher_->layout()->dim()[i]; ++j) {
                            for (Size k=0; k<x_.size(); ++k) {
                                Size index = j*ySpacing + k*xSpacing;
                                tmp[k] = aCopy[index];
                            }
                            jump.apply(tmp, jumped);
                            for (Size k=0; k<x_.size(); ++k)
                                a[j*ySpacing + k*xSpacing] = jumped[k];
                        }
                    }
                }
            }
        }
    }
}
