/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Intraday equity-option clock: time measured from now to each date's close.

 QuantLib turns a date into a time only through a day counter,
 dayCounter().yearFraction(referenceDate(), date), so an option expiring
 at today's close has no time left when today is the reference date.  The
 equity-option clock instead takes the previous session's date A as its
 reference and a quote holding "now" as years from A (midnight, Actual/365
 Fixed).  A date d stands for its close, c_d years after its midnight, and

   yearFraction(A, d) = ACT(A, d) + c_d - now,   yearFraction(A, A) = 0

 is the calendar time from now to that close; A itself is "now" (the start
 of an American exercise).  Moving the quote moves every time on the clock.

 IntradayYieldTermStructure presents a base discount curve on that clock:
 discount(t) is the base curve's discount factor from now to now + t, the
 base curve's reference date being at or before today's midnight.
*/

#ifndef quantlib_intraday_equity_clock_hpp
#define quantlib_intraday_equity_clock_hpp

#include <ql/handle.hpp>
#include <ql/quote.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/time/daycounter.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/utilities/null.hpp>
#include <cmath>
#include <sstream>
#include <utility>

namespace QuantLib {

    //! Actual/365 (Fixed) time from now (a quote) to each date's close
    /*! \p now is years from \p anchor's midnight to the current instant;
        \p close is a date's close in years after its midnight, and
        \p session / \p sessionClose override it for one date (an early
        close).  Only times measured from \p anchor move with the quote;
        between two later dates the count is plain Actual/365 between
        their closes. */
    class IntradayActual365Fixed : public DayCounter {
      private:
        class Impl : public DayCounter::Impl {
          public:
            Impl(const Date& anchor, Handle<Quote> now, Real close, const Date& session,
                 Real sessionClose)
            : anchor_(anchor), now_(std::move(now)), close_(close), session_(session),
              sessionClose_(sessionClose) {}
            std::string name() const override {
                std::ostringstream out;
                out << "Actual/365 (Fixed) from now, anchor " << anchor_;
                return out.str();
            }
            Date::serial_type dayCount(const Date& d1, const Date& d2) const override {
                return d2 - d1;
            }
            Time yearFraction(const Date& d1, const Date& d2, const Date&, const Date&)
                const override {
                return at(d2) - at(d1);
            }

          private:
            //! The instant d stands for, in years from the anchor's midnight.
            Time at(const Date& d) const {
                if (d == anchor_)
                    return now_->value();
                const Real c = d == session_ ? sessionClose_ : close_;
                return (d - anchor_) / 365.0 + c;
            }
            Date anchor_;
            Handle<Quote> now_;
            Real close_;
            Date session_;
            Real sessionClose_;
        };

      public:
        IntradayActual365Fixed(const Date& anchor,
                               const Handle<Quote>& now,
                               Real close,
                               const Date& session = Date(),
                               Real sessionClose = Null<Real>())
        : DayCounter(ext::shared_ptr<DayCounter::Impl>(
              new Impl(anchor, now, close, session,
                       sessionClose == Null<Real>() ? close : sessionClose))) {
            QL_REQUIRE(!now.empty(), "IntradayActual365Fixed: empty now handle");
            QL_REQUIRE(close >= 0.0 && close < 1.0 / 365.0,
                       "IntradayActual365Fixed: close " << close << " is not within a day");
        }
    };

    //! A base discount curve on the intraday equity-option clock
    /*! The reference date is the clock's anchor and the day counter the
        clock itself; discount(t) = base(n + t) / base(n), with n the
        current instant on the base curve's clock (its reference date at or
        before the instant).  The base day counter must be Actual/N. */
    class IntradayYieldTermStructure : public YieldTermStructure {
      public:
        IntradayYieldTermStructure(Handle<YieldTermStructure> base,
                                   const Date& anchor,
                                   Handle<Quote> now,
                                   const DayCounter& clock)
        : YieldTermStructure(anchor, NullCalendar(), clock), base_(std::move(base)),
          now_(std::move(now)) {
            QL_REQUIRE(!base_.empty(), "IntradayYieldTermStructure: empty base curve handle");
            QL_REQUIRE(!now_.empty(), "IntradayYieldTermStructure: empty now handle");
            const Date d0(1, January, 2000);
            const DayCounter baseDc = base_->dayCounter();
            // Base-curve time per year of Actual/365 Fixed.
            factor_ = baseDc.yearFraction(d0, d0 + 36500) / 100.0;
            for (Integer n : {1, 17, 400, 3653}) {
                const Real expected = factor_ * n / 365.0;
                QL_REQUIRE(std::fabs(baseDc.yearFraction(d0, d0 + n) - expected) <=
                               1e-12 * std::max<Real>(1.0, expected),
                           "IntradayYieldTermStructure: base day counter "
                               << baseDc.name() << " is not Actual/N");
            }
            registerWith(base_);
            registerWith(now_);
        }
        Date maxDate() const override { return base_->maxDate(); }
        //! The current instant on the base curve's clock.
        Time baseNow() const {
            const Time n =
                (now_->value() - (base_->referenceDate() - referenceDate()) / 365.0) *
                factor_;
            QL_REQUIRE(n >= 0.0, "IntradayYieldTermStructure: now is before the base curve's "
                                 "reference date "
                                     << base_->referenceDate());
            return n;
        }

      protected:
        DiscountFactor discountImpl(Time t) const override {
            const Time n = baseNow();
            return base_->discount(n + t * factor_, true) / base_->discount(n, true);
        }

      private:
        Handle<YieldTermStructure> base_;
        Handle<Quote> now_;
        Real factor_;
    };

}

#endif
