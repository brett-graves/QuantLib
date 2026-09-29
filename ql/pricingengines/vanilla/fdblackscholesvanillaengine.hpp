/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008 Ralph Schreyer
 Copyright (C) 2008, 2009 Klaus Spanderen

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

/*! \file fdblackscholesvanillaengine.hpp
    \brief Finite-differences Black Scholes vanilla option engine
*/

#ifndef quantlib_fd_black_scholes_vanilla_engine_hpp
#define quantlib_fd_black_scholes_vanilla_engine_hpp

#include <ql/pricingengine.hpp>
#include <ql/pricingengines/vanilla/cashdividendeuropeanengine.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <tuple>
#include <vector>

namespace QuantLib {

    class Fdm1dMesher;
    class FdmQuantoHelper;
    class GeneralizedBlackScholesProcess;

    //! Finite-differences Black Scholes vanilla option engine
    /*! Optional mesh and time-grid controls (defaults reproduce the
        classic engine exactly):

        - \p mesherScaleFactor, \p mesherEps: the ln(S) range of the
          FdmBlackScholesMesher, i.e. the forward's span widened by
          sigma(T, K) sqrt(T) N^{-1}(1 - eps) scaleFactor on each side.
          Local-vol surfaces whose wings carry much higher vol than
          sigma(T, K) need a wider range than the default 1.5.
        - \p spotConcentrationDensity: when given, the mesh concentrates
          at the (dividend-adjusted) spot as well as at the strike, both
          through one multi-point Concentrating1dMesher over the same
          range; densities are relative to the range width, as in
          Concentrating1dMesher (the strike keeps 0.1).  The spot is read
          at every calculate(), so a cached engine follows spot moves.
        - \p concentrationPoints: further (level, density, required)
          points; the level is in units of the underlying (not ln S),
          density as above, required forces the level onto the grid.
          Points outside the range are ignored.
        - \p stoppingTimes: extra times (process time, the risk-free
          curve's clock) the rollback must stop at, joined into the
          step-condition composite.  Local-vol surfaces interpolated
          linearly in total variance between expiry pillars have local
          vol discontinuous in time at the pillars; stopping there keeps
          any time step from straddling one.  Times outside
          (0, maturity) are ignored.

        \ingroup vanillaengines

        \test the correctness of the returned value is tested by
              reproducing results available in web/literature
              and comparison with Black pricing.
    */
    class FdBlackScholesVanillaEngine : public VanillaOption::engine {
      public:
        enum CashDividendModel {
            Spot = CashDividendEuropeanEngine::Spot,
            Escrowed = CashDividendEuropeanEngine::Escrowed
        };

        explicit FdBlackScholesVanillaEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess>,
            Size tGrid = 100,
            Size xGrid = 100,
            Size dampingSteps = 0,
            const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas(),
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            CashDividendModel cashDividendModel = Spot,
            Real mesherScaleFactor = 1.5,
            Real mesherEps = 0.0001,
            Real spotConcentrationDensity = Null<Real>(),
            std::vector<std::tuple<Real, Real, bool> > concentrationPoints = {},
            std::vector<Time> stoppingTimes = {});

        FdBlackScholesVanillaEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess>,
            DividendSchedule dividends,
            Size tGrid = 100,
            Size xGrid = 100,
            Size dampingSteps = 0,
            const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas(),
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            CashDividendModel cashDividendModel = Spot,
            Real mesherScaleFactor = 1.5,
            Real mesherEps = 0.0001,
            Real spotConcentrationDensity = Null<Real>(),
            std::vector<std::tuple<Real, Real, bool> > concentrationPoints = {},
            std::vector<Time> stoppingTimes = {});

        FdBlackScholesVanillaEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess>,
            ext::shared_ptr<FdmQuantoHelper> quantoHelper,
            Size tGrid = 100,
            Size xGrid = 100,
            Size dampingSteps = 0,
            const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas(),
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            CashDividendModel cashDividendModel = Spot,
            Real mesherScaleFactor = 1.5,
            Real mesherEps = 0.0001,
            Real spotConcentrationDensity = Null<Real>(),
            std::vector<std::tuple<Real, Real, bool> > concentrationPoints = {},
            std::vector<Time> stoppingTimes = {});

        FdBlackScholesVanillaEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess>,
            DividendSchedule dividends,
            ext::shared_ptr<FdmQuantoHelper> quantoHelper,
            Size tGrid = 100,
            Size xGrid = 100,
            Size dampingSteps = 0,
            const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas(),
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            CashDividendModel cashDividendModel = Spot,
            Real mesherScaleFactor = 1.5,
            Real mesherEps = 0.0001,
            Real spotConcentrationDensity = Null<Real>(),
            std::vector<std::tuple<Real, Real, bool> > concentrationPoints = {},
            std::vector<Time> stoppingTimes = {});

        void calculate() const override;

        //! Local-vol grid points priced with illegalLocalVolOverwrite in
        //! the last calculate() (0 without local vol).
        Size illegalLocalVolCount() const { return illegalLocalVolCount_; }

      private:
        ext::shared_ptr<Fdm1dMesher> equityMesher(
            Time maturity, Real strike,
            const DividendSchedule& dividendSchedule,
            Real spotAdjustment) const;

        ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        DividendSchedule dividends_;
        Size tGrid_, xGrid_, dampingSteps_;
        FdmSchemeDesc schemeDesc_;
        bool localVol_;
        Real illegalLocalVolOverwrite_;
        ext::shared_ptr<FdmQuantoHelper> quantoHelper_;
        CashDividendModel cashDividendModel_;
        Real mesherScaleFactor_, mesherEps_, spotConcentrationDensity_;
        std::vector<std::tuple<Real, Real, bool> > concentrationPoints_;
        std::vector<Time> stoppingTimes_;
        mutable Size illegalLocalVolCount_ = 0;
    };


    class MakeFdBlackScholesVanillaEngine {
      public:
        explicit MakeFdBlackScholesVanillaEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess> process);

        MakeFdBlackScholesVanillaEngine& withQuantoHelper(
            const ext::shared_ptr<FdmQuantoHelper>& quantoHelper);

        MakeFdBlackScholesVanillaEngine& withTGrid(Size tGrid);
        MakeFdBlackScholesVanillaEngine& withXGrid(Size xGrid);
        MakeFdBlackScholesVanillaEngine& withDampingSteps(
            Size dampingSteps);

        MakeFdBlackScholesVanillaEngine& withFdmSchemeDesc(
            const FdmSchemeDesc& schemeDesc);

        MakeFdBlackScholesVanillaEngine& withLocalVol(bool localVol);
        MakeFdBlackScholesVanillaEngine& withIllegalLocalVolOverwrite(
            Real illegalLocalVolOverwrite);

        MakeFdBlackScholesVanillaEngine& withCashDividends(
            const std::vector<Date>& dividendDates,
            const std::vector<Real>& dividendAmounts);

        MakeFdBlackScholesVanillaEngine& withCashDividendModel(
            FdBlackScholesVanillaEngine::CashDividendModel cashDividendModel);

        MakeFdBlackScholesVanillaEngine& withMesherScaleFactor(
            Real scaleFactor);
        MakeFdBlackScholesVanillaEngine& withMesherEps(Real eps);
        MakeFdBlackScholesVanillaEngine& withSpotConcentration(
            Real density);
        MakeFdBlackScholesVanillaEngine& withConcentrationPoints(
            const std::vector<std::tuple<Real, Real, bool> >& points);
        MakeFdBlackScholesVanillaEngine& withStoppingTimes(
            const std::vector<Time>& stoppingTimes);

        operator ext::shared_ptr<PricingEngine>() const;
      private:
        ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        DividendSchedule dividends_;
        Size tGrid_ = 100, xGrid_ = 100, dampingSteps_ = 0;
        ext::shared_ptr<FdmSchemeDesc> schemeDesc_;
        bool localVol_ = false;
        Real illegalLocalVolOverwrite_;
        ext::shared_ptr<FdmQuantoHelper> quantoHelper_;
        FdBlackScholesVanillaEngine::CashDividendModel cashDividendModel_ = FdBlackScholesVanillaEngine::Spot;
        Real mesherScaleFactor_ = 1.5, mesherEps_ = 0.0001;
        Real spotConcentrationDensity_;
        std::vector<std::tuple<Real, Real, bool> > concentrationPoints_;
        std::vector<Time> stoppingTimes_;
    };

}

#endif
