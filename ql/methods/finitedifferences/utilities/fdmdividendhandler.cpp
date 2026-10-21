/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008, 2009 Ralph Schreyer
 Copyright (C) 2008 Klaus Spanderen
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

#include <ql/time/daycounter.hpp>
#include <ql/cashflows/dividend.hpp>
#include <ql/math/interpolations/linearinterpolation.hpp>
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
      schedule_(schedule),
      mesher_(mesher),
      equityDirection_(equityDirection) {

        dividends_.reserve(schedule.size());
        dividendDates_.reserve(schedule.size());
        dividendTimes_.reserve(schedule.size());
        for (const auto& iter : schedule) {
            // a fractional dividend without a nominal has no fixed amount;
            // its drop is computed from the grid values in applyTo
            auto fractional = ext::dynamic_pointer_cast<FractionalDividend>(iter);
            dividends_.push_back(fractional != nullptr && fractional->nominal() == Null<Real>()
                                     ? Null<Real>()
                                     : iter->amount());
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
            const ext::shared_ptr<Dividend>& dividend = schedule_[iter - dividendTimes_.begin()];

            // the ex-dividend spot for each grid value; a fractional
            // dividend pays a fraction of the spot at the ex-date
            Array exDividend(x_.size());
            for (Size k=0; k<x_.size(); ++k)
                exDividend[k] = std::max(x_[0], x_[k] - dividend->amount(x_[k]));

            if (mesher_->layout()->dim().size() == 1) {
                LinearInterpolation interp(x_.begin(), x_.end(), aCopy.begin());
                for (Size k=0; k<x_.size(); ++k) {
                    a[k] = interp(exDividend[k], true);
                }
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
                            LinearInterpolation interp(x_.begin(), x_.end(),
                                                       tmp.begin());
                            for (Size k=0; k<x_.size(); ++k) {
                                Size index = j*ySpacing + k*xSpacing;
                                a[index] = interp(exDividend[k], true);
                            }
                        }
                    }
                }
            }
        }
    }
}
