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

/*! \file shiftedtimeblackvol.hpp
    \brief Black variance of a base surface read a fixed time earlier
*/

#ifndef quantlib_shifted_time_black_vol_hpp
#define quantlib_shifted_time_black_vol_hpp

#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <utility>

namespace QuantLib {

    //! Black variance of a base surface read \p offset years earlier.
    /*! An option whose exercise date stands for an earlier instant than
        that date's close: an A.M.-settled index option (SPX) settles on
        the opening quotation of its expiry date, about 6.5 hours before
        the close the date would otherwise mean.  Engines turn the
        exercise date into a time through the vol's day counter; this
        wrapper hands them the base surface's total variance at that time
        less \p offset, so the option reads its own pillar.  Reference
        date, day counter and calendar are the base's, and so are its
        notifications. */
    class ShiftedTimeBlackVol : public BlackVarianceTermStructure {
      public:
        ShiftedTimeBlackVol(Handle<BlackVolTermStructure> base, Time offset)
        : BlackVarianceTermStructure(base->businessDayConvention(), base->dayCounter()),
          base_(std::move(base)), offset_(offset) {
            QL_REQUIRE(offset_ >= 0.0, "offset " << offset_ << " must not be negative");
            registerWith(base_);
        }
        //! \name TermStructure interface
        //@{
        const Date& referenceDate() const override { return base_->referenceDate(); }
        DayCounter dayCounter() const override { return base_->dayCounter(); }
        Calendar calendar() const override { return base_->calendar(); }
        Natural settlementDays() const override { return base_->settlementDays(); }
        Date maxDate() const override { return base_->maxDate(); }
        //@}
        //! \name VolatilityTermStructure interface
        //@{
        Real minStrike() const override { return base_->minStrike(); }
        Real maxStrike() const override { return base_->maxStrike(); }
        //@}
        Time offset() const { return offset_; }

      protected:
        Real blackVarianceImpl(Time t, Real strike) const override {
            const Time s = t - offset_;
            QL_REQUIRE(s > 0.0, "time " << t << " is not after the offset " << offset_
                                        << ": the option has settled");
            return base_->blackVariance(s, strike, true);
        }

      private:
        Handle<BlackVolTermStructure> base_;
        Time offset_;
    };

}

#endif
