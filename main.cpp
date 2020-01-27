
#ifndef __CLING__
        #pragma error "Must be runed under ROOT or ACLiC"
#endif

// Enable to display plots of layer activity
//#define SHOW_MODE

#include <Etaler/Etaler.hpp>
#include <Etaler/Encoders/GridCell1d.hpp>
#include <Etaler/Encoders/GridCell2d.hpp>
#include <Etaler/Encoders/Category.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>
#include <Etaler/Algorithms/Anomaly.hpp>
// Load OpenCL and Etaler
#pragma cling load("OpenCL")
#pragma cling load("/usr/local/lib/libEtaler.so")

#include <TPython.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TGraph.h>
#include <TROOT.h>
#include <TLegend.h>
#include <TAxis.h>

using namespace std;
using namespace et;

#include "layer_abstractions.hpp"

Shape channel_shape = {1024};

struct HTMAgent
{
	static constexpr int cells_per_column = 12;
	SP sensor;
	ATM l4;
	TM l3;
	TM l5;
	TM d1;
	TM d2;
	ATM motor;
	SP motor_pooling;

	Tensor d1_active, d1_old_active;
	Tensor d2_active, d2_old_active;

	HTMAgent()
	{
		channel_shape  = {1024};
		// Layer initalization
		sensor = SP(encode(0,0,0,0).shape(), channel_shape);
		l4 = make_atm();
		l3 = make_tm();
		l5 = make_tm();
		d1 = make_tm();
		d2 = make_tm();
		motor = make_atm();
		motor_pooling = SP(channel_shape, {128, 2});

		sensor.setGlobalDensity(0.08);
		motor_pooling.setGlobalDensity(0.08);
		motor_pooling.setBoostingFactor(0.1);

		reset();
	}

	Tensor encode(float v0, float v1, float v2, float v3) const
	{
		Tensor s0 = encoder::gridCell2d({v0, v2});
		Tensor s1 = encoder::gridCell2d({v1, v3});
		Tensor s2 = encoder::gridCell2d({(float)(v3 > 0), float(v1 > 0)});
		return concat({s0, s1, s2});
	}

	int compute(float v0, float v1, float v2, float v3, bool learn)
	{
		auto x = encode(v0, v1, v2, v3);
		auto sensor_out = sensor.compute(x, learn);
		l4.compute(sensor_out, l5.activeCells(), learn);
		l3.compute(l4.activeColumns(), learn);
		l5.compute(l4.activeColumns(), l3.predictiveCells(), learn);
		tie(d1_old_active, d2_old_active) = tuple(d1.activeCells(), d2.activeCells());
		d1.compute(l4.activeColumns(), l5.predictiveCells(), false);
		d2.compute(l4.activeColumns(), l5.predictiveCells(), false);
		tie(d1_active, d2_active) = tuple(d1.activeCells(), d2.activeCells());
		motor.compute(l5.activeColumns(), d1.activeCells() && (!d2.activeCells()), learn);
		auto motor_out = motor_pooling.compute(motor.activeColumns(), learn);
		auto motor_activation = motor_out.sum(0).toHost<int>();
		if(motor_activation[0] > motor_activation[1])
			return 0;
		return 1;
	}

	void learn(float reward)
	{
		if(reward > 0) {
			// Positive reinforcment
			d1.learn(d1_active, d1_old_active);
			learnCorrilation(d2_old_active, d2_active, d2.connections_, d2.permanences_, -d2.permanenceInc(), 0);
		}
		else {
			d2.learn(d2_active, d2_old_active);
			learnCorrilation(d1_old_active, d1_active, d1.connections_, d1.permanences_, -d1.permanenceInc(), 0);
		}
	}

	void reset()
	{
		l4.reset();
		l3.reset();
		l5.reset();
		d1.reset();
		d2.reset();
		motor.reset();
	}

	// Helper
	ATM make_atm(Shape input_shape = channel_shape, intmax_t cells = cells_per_column, Shape apical_shape = channel_shape+cells_per_column) const
	{
		return ATM(input_shape, cells, apical_shape);
	}

	TM make_tm(Shape input_shape = channel_shape, intmax_t cells = cells_per_column)
	{
		return TM(input_shape, cells);
	}
};

int main()
{
	// gStyle->SetCanvasPreferGL(true); // Do NOT turn on. This messes up Gym's rendering
	//setDefaultBackend(std::make_shared<OpenCLBackend>());
	TPython::ExecScript("main.py"); 
}
