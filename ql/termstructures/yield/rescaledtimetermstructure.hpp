/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Rescaled-time yield term structure: the same discount curve presented
 on another fixed-denominator Actual day counter.

 A process's clock is its risk-free curve's day counter.  When that is
 Actual/360 (money-market convention) and the volatility surface is
 Actual/365 Fixed, finite-difference engines step the PDE on the curve's
 clock and query the vol/local-vol surface at the wrong times.  Wrapping
 the process curves in this class puts every term structure the engine
 touches on the surface's clock while leaving every discount factor at
 every date unchanged:

   discount_this(t) = discount_base(t * factor),
   factor = (days per year of this day counter) / (days per year of base)

 which is exact for any pair of Actual/N day counters, since both measure
 time as actual days over a constant.
*/

#ifndef quantlib_rescaled_time_term_structure_hpp
#define quantlib_rescaled_time_term_structure_hpp

#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/time/date.hpp>
#include <cmath>
#include <utility>

namespace QuantLib {

    //! Yield term structure presenting a base curve on another Actual/N day counter
    /*! Discount factors at dates are those of the base curve; only the
        time axis (and therefore anything expressed per unit time, such as
        zero or forward rates) is on the new day counter.  The reference
        date, calendar and settlement days follow the base curve.

        Both day counters must count actual days over a fixed
        denominator (Actual/360, Actual/365 Fixed, Actual/364, ...);
        the constructor checks that the base-to-new time ratio is the
        same over a short and a long horizon and refuses otherwise.

        \note This term structure remains linked to the base: any change
              in the base (bumps, relinking) is reflected here.

        \ingroup yieldtermstructures
    */
    class RescaledTimeYieldTermStructure : public YieldTermStructure {
      public:
        RescaledTimeYieldTermStructure(Handle<YieldTermStructure> base,
                                       const DayCounter& dayCounter)
        : YieldTermStructure(dayCounter), base_(std::move(base)) {
            QL_REQUIRE(!base_.empty(), "base curve handle must not be empty");
            const Date d0(1, January, 2000);
            const DayCounter baseDc = base_->dayCounter();
            factor_ = baseDc.yearFraction(d0, d0 + 36500)
                      / dayCounter.yearFraction(d0, d0 + 36500);
            for (Integer n : {1, 17, 400, 3653}) {
                Real expected = factor_ * dayCounter.yearFraction(d0, d0 + n);
                QL_REQUIRE(std::fabs(baseDc.yearFraction(d0, d0 + n) - expected)
                               <= 1e-12 * std::max<Real>(1.0, expected),
                           "RescaledTimeYieldTermStructure: " << baseDc.name()
                           << " and " << dayCounter.name()
                           << " are not both Actual/N day counters");
            }
            registerWith(base_);
        }
        //! \name TermStructure interface
        //@{
        const Date& referenceDate() const override { return base_->referenceDate(); }
        Calendar calendar() const override { return base_->calendar(); }
        Natural settlementDays() const override { return base_->settlementDays(); }
        Date maxDate() const override { return base_->maxDate(); }
        //@}
        //! Base-curve time per unit of this curve's time.
        Real factor() const { return factor_; }

      protected:
        DiscountFactor discountImpl(Time t) const override {
            return base_->discount(t * factor_, true);
        }

      private:
        Handle<YieldTermStructure> base_;
        Real factor_;
    };

}

#endif
