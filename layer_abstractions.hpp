#pragma once

#include <Etaler/Etaler.hpp>
#include <Etaler/Encoders/Category.hpp>
#include <Etaler/Backends/OpenCLBackend.hpp>
#include <Etaler/Algorithms/SpatialPooler.hpp>
#include <Etaler/Algorithms/TemporalMemory.hpp>

#include <tuple>

using namespace et;
using namespace std;

struct SP : public et::SpatialPooler
{
	SP(Shape input_shape, Shape output_shape)
		: et::SpatialPooler(input_shape, output_shape)
	{}

	SP() = default;

	Tensor compute(const Tensor& x, bool learning = false)
	{
		Tensor y = SpatialPooler::compute(x);
		if(learning)
			learn(x, y);
		return y;
	}
};

struct TM : public et::TemporalMemory
{
	TM() = default;
	TM(Shape input_shape, intmax_t cells_per_column)
		: TemporalMemory(input_shape, cells_per_column, 4096)
	{
		predictive_cells_ = et::zeros(input_shape+cells_per_column, DType::Bool);
		active_cells_ = et::zeros(input_shape+cells_per_column, DType::Bool);
	}

	void compute(const Tensor& x, bool learning)
	{
		auto [pred, active] = TemporalMemory::compute(x, predictive_cells_);
		if(learning)
			TemporalMemory::learn(active, active_cells_);
		tie(predictive_cells_, active_cells_) = tuple(pred, active);
	}

	// Allow external control of the context
	void compute(const Tensor& x, const Tensor& context, bool learning)
	{
		predictive_cells_ = context;
		compute(x, learning);
	}

	Tensor predictiveCells() const
	{
		return predictive_cells_;
	}

	Tensor activeCells() const
	{
		return active_cells_;
	}

	Tensor activeColumns() const
	{
		return active_cells_.sum(-1, DType::Bool);
	}

	void reset()
	{
		predictive_cells_ = et::zeros_like(predictive_cells_);
		active_cells_ = et::zeros_like(active_cells_);
	}

	void decay()
	{
		decaySynapses(connections_, permanences_, active_threshold_);
	}

	Tensor predictive_cells_;
	Tensor active_cells_;
};

struct ApicalTemporalMemroy : public et::TemporalMemory
{
	ApicalTemporalMemroy() = default;
	ApicalTemporalMemroy(Shape input_shape, intmax_t cells_per_column, Shape apical_shape)
		: TemporalMemory(input_shape, cells_per_column, 4096)
		, apical_synapses_(apical_shape, input_shape)
	{

	}

	std::tuple<Tensor, Tensor, Tensor> compute(const Tensor& x, const Tensor& apical, const Tensor& last_state)
	{
		et_assert(x.dimentions()  == 1);//This is a 1D implementation
		assert(apical.shape() == apical_synapses_.input_shape_);

		//Feed forward TM predictions
		auto [pred, active] = TemporalMemory::compute(x, last_state);

		//Apical feedback
		Tensor feedbacks = apical_synapses_.compute(apical);
		Tensor feedbacks_reshaped = feedbacks.reshape(feedbacks.shape() + 1);

		//Cells only  predict if it feedback is active when there is more than 1 cells active in a column
		auto s = pred.sum(1);
		s = s.reshape({(intmax_t)s.size(), 1});
		pred = pred && ((s && feedbacks_reshaped) || (s<=1 && pred));

		return {pred, active, feedbacks};
	}

	void learn(const Tensor& active_cells, const Tensor& apical, const Tensor& last_active)
	{
		assert(apical.shape() == apical_synapses_.input_shape_);
		//Let the distal synapses grow and learn
		TemporalMemory::learn(active_cells, last_active);

		//Let the apical synapses learn
		// apical_synapses_.learn(last_active, apical);
	}

	SpatialPooler apical_synapses_;
};

struct ATM : public ApicalTemporalMemroy
{
	ATM() = default;
	ATM(Shape input_shape, intmax_t cells_per_column, Shape apical_shape)
		: ApicalTemporalMemroy(input_shape, cells_per_column, apical_shape)
	{
		predictive_cells_ = et::zeros(input_shape+cells_per_column, DType::Bool);
		active_cells_ = et::zeros(input_shape+cells_per_column, DType::Bool);
	}
	
	void compute(const Tensor& x, const Tensor& feedback, bool learning)
	{
		auto [pred, active, apical] = ApicalTemporalMemroy::compute(x, feedback, predictive_cells_);
		if(learning)
			ApicalTemporalMemroy::learn(active, apical, active_cells_);
		tie(predictive_cells_, active_cells_) = tuple(pred, active);
	}

	Tensor predictiveCells() const
	{
		return predictive_cells_;
	}

	Tensor activeCells() const
	{
		return active_cells_;
	}

	Tensor activeColumns() const
	{
		return active_cells_.sum(-1, DType::Bool);
	}

	void reset()
	{
		predictive_cells_ = et::zeros_like(predictive_cells_);
		active_cells_ = et::zeros_like(active_cells_);
	}

	void decay()
	{
		decaySynapses(connections_, permanences_, 0.0001);
	}


	Tensor predictive_cells_;
	Tensor active_cells_;
};