//*************************************************************************
// semiring
//*************************************************************************

namespace gtsam {

#include <gtsam/semiring/SemiringFactor.h>
virtual class SemiringFactor : gtsam::Factor {
  bool equals(const gtsam::SemiringFactor& other, double tol = 1e-9) const;

  gtsam::SemiringFactor* multiply(const gtsam::SemiringFactor& other) const;
  pair<gtsam::SemiringConditional*, gtsam::SemiringFactor*> eliminate(
      const gtsam::Ordering& frontalKeys) const;
  gtsam::SemiringFactor* sum(const gtsam::Ordering& frontalKeys) const;
  double expectation() const;
};

#include <gtsam/semiring/SemiringConditional.h>
virtual class SemiringConditional : gtsam::SemiringFactor {
  void print(string s = "SemiringConditional",
             const gtsam::KeyFormatter& formatter =
                 gtsam::DefaultKeyFormatter) const;

  const gtsam::SemiringFactor::shared_ptr& factor() const;
  gtsam::Key firstFrontalKey() const;
  size_t nrFrontals() const;
  size_t nrParents() const;
};

#include <gtsam/semiring/SemiringDiscreteFactor.h>
virtual class SemiringDiscreteFactor : gtsam::SemiringFactor {
  SemiringDiscreteFactor();
  SemiringDiscreteFactor(const gtsam::DecisionTreeFactor& probability);
  SemiringDiscreteFactor(const gtsam::DecisionTreeFactor& probability,
                         const gtsam::DecisionTreeFactor& value);
  static gtsam::SemiringDiscreteFactor Reward(
      const gtsam::DecisionTreeFactor& reward);
  static gtsam::SemiringDiscreteFactor FromChannels(
      const gtsam::DecisionTreeFactor& probability,
      const gtsam::DecisionTreeFactor& weightedValue);

  void print(string s = "SemiringDiscreteFactor",
             const gtsam::KeyFormatter& formatter =
                 gtsam::DefaultKeyFormatter) const;

  const gtsam::DecisionTreeFactor& probability() const;
  const gtsam::DecisionTreeFactor& weightedValue() const;
  gtsam::DecisionTreeFactor value() const;
  gtsam::DiscreteKeys discreteKeys() const;
  pair<double, double> evaluate(const gtsam::DiscreteValues& values) const;

  gtsam::SemiringDiscreteFactor operator*(
      const gtsam::SemiringDiscreteFactor& other) const;
  gtsam::SemiringDiscreteFactor operator/(
      const gtsam::SemiringDiscreteFactor& other) const;
};

#include <gtsam/semiring/SemiringDiscreteConditional.h>
virtual class SemiringDiscreteConditional : gtsam::SemiringConditional {
  SemiringDiscreteConditional();
  SemiringDiscreteConditional(const gtsam::SemiringDiscreteFactor& joint,
                              const gtsam::SemiringDiscreteFactor& marginal,
                              const gtsam::Ordering& frontalKeys);

  const gtsam::SemiringDiscreteFactor& table() const;
  gtsam::DiscreteConditional probability() const;
  gtsam::DecisionTreeFactor surprise() const;
  pair<double, double> evaluate(const gtsam::DiscreteValues& values) const;
};

#include <gtsam/semiring/SemiringGaussianFactor.h>
virtual class SemiringGaussianFactor : gtsam::SemiringFactor {
  SemiringGaussianFactor();
  SemiringGaussianFactor(const gtsam::GaussianFactor* gaussian);
  SemiringGaussianFactor(const gtsam::GaussianFactorGraph& gaussian);
  SemiringGaussianFactor(const gtsam::GaussianFactorGraph& gaussian,
                         const gtsam::HessianFactor* value);
  static gtsam::SemiringGaussianFactor Reward(
      const gtsam::GaussianFactor& quadratic);
  static gtsam::SemiringGaussianFactor Cost(
      const gtsam::GaussianFactor& quadratic);

  void print(string s = "SemiringGaussianFactor",
             const gtsam::KeyFormatter& formatter =
                 gtsam::DefaultKeyFormatter) const;

  const gtsam::GaussianFactorGraph& gaussian() const;
  const gtsam::HessianFactor::shared_ptr& value() const;
  double error(const gtsam::VectorValues& x) const;
  double value(const gtsam::VectorValues& x) const;
};

#include <gtsam/semiring/SemiringGaussianConditional.h>
virtual class SemiringGaussianConditional : gtsam::SemiringConditional {
  SemiringGaussianConditional();
  SemiringGaussianConditional(const gtsam::GaussianConditional* conditional);
  SemiringGaussianConditional(const gtsam::GaussianConditional* conditional,
                              const gtsam::HessianFactor* surprise);

  const gtsam::GaussianConditional::shared_ptr& conditional() const;
  const gtsam::HessianFactor::shared_ptr& surprise() const;
  double surprise(const gtsam::VectorValues& x) const;
};

#include <gtsam/semiring/SemiringBayesNet.h>
class SemiringBayesNet {
  SemiringBayesNet();
  void push_back(const gtsam::SemiringConditional* conditional);

  bool empty() const;
  size_t size() const;
  gtsam::KeySet keys() const;
  const gtsam::SemiringConditional* at(size_t i) const;

  void print(string s = "SemiringBayesNet\n",
             const gtsam::KeyFormatter& formatter =
                 gtsam::DefaultKeyFormatter) const;
  bool equals(const gtsam::SemiringBayesNet& bn, double tol = 1e-9) const;
};

#include <gtsam/semiring/SemiringBayesTree.h>
class SemiringBayesTreeClique {
  SemiringBayesTreeClique();
  SemiringBayesTreeClique(const gtsam::SemiringConditional* conditional);
  const gtsam::SemiringConditional::shared_ptr& conditional() const;
  bool isRoot() const;
  size_t nrChildren() const;
  const gtsam::SemiringBayesTreeClique* operator[](size_t i) const;
  void print(string s = "SemiringBayesTreeClique",
             const gtsam::KeyFormatter& keyFormatter =
                 gtsam::DefaultKeyFormatter) const;
};

class SemiringBayesTree {
  SemiringBayesTree();

  void print(string s = "SemiringBayesTree\n",
             const gtsam::KeyFormatter& keyFormatter =
                 gtsam::DefaultKeyFormatter) const;
  bool equals(const gtsam::SemiringBayesTree& other, double tol = 1e-9) const;

  size_t size() const;
  bool empty() const;
  const gtsam::SemiringBayesTreeClique* operator[](gtsam::Key j) const;

  std::shared_ptr<gtsam::SemiringConditional> marginalFactor(
      gtsam::Key j,
      const gtsam::SemiringFactorGraph::Eliminate& function =
          gtsam::SemiringFactorGraph::Eliminate(
              gtsam::SemiringFactorGraph::EliminationTraitsType::DefaultEliminate))
      const;
};

#include <gtsam/semiring/SemiringFactorGraph.h>
pair<gtsam::SemiringConditional*, gtsam::SemiringFactor*> EliminateSemiring(
    const gtsam::SemiringFactorGraph& factors,
    const gtsam::Ordering& frontalKeys);

#include <gtsam/inference/EliminateableFactorGraph.h>
class SemiringFactorGraph {
  std::shared_ptr<gtsam::SemiringBayesNet> eliminateSequential(
      gtsam::SemiringFactorGraph::OptionalOrderingType orderingType = std::nullopt,
      const gtsam::SemiringFactorGraph::Eliminate& function =
          gtsam::SemiringFactorGraph::Eliminate(
              gtsam::SemiringFactorGraph::EliminationTraitsType::DefaultEliminate),
      gtsam::SemiringFactorGraph::OptionalVariableIndex variableIndex = std::nullopt)
      const;
  std::shared_ptr<gtsam::SemiringBayesNet> eliminateSequential(
      const gtsam::Ordering& ordering,
      const gtsam::SemiringFactorGraph::Eliminate& function =
          gtsam::SemiringFactorGraph::Eliminate(
              gtsam::SemiringFactorGraph::EliminationTraitsType::DefaultEliminate),
      gtsam::SemiringFactorGraph::OptionalVariableIndex variableIndex = std::nullopt)
      const;
  pair<std::shared_ptr<gtsam::SemiringBayesNet>,
       std::shared_ptr<gtsam::SemiringFactorGraph>>
  eliminatePartialSequential(
      const gtsam::Ordering& ordering,
      const gtsam::SemiringFactorGraph::Eliminate& function =
          gtsam::SemiringFactorGraph::Eliminate(
              gtsam::SemiringFactorGraph::EliminationTraitsType::DefaultEliminate),
      gtsam::SemiringFactorGraph::OptionalVariableIndex variableIndex = std::nullopt)
      const;
  pair<std::shared_ptr<gtsam::SemiringBayesNet>,
       std::shared_ptr<gtsam::SemiringFactorGraph>>
  eliminatePartialSequential(
      const gtsam::KeyVector& variables,
      const gtsam::SemiringFactorGraph::Eliminate& function =
          gtsam::SemiringFactorGraph::Eliminate(
              gtsam::SemiringFactorGraph::EliminationTraitsType::DefaultEliminate),
      gtsam::SemiringFactorGraph::OptionalVariableIndex variableIndex = std::nullopt)
      const;
  std::shared_ptr<gtsam::SemiringBayesTree> eliminateMultifrontal(
      gtsam::SemiringFactorGraph::OptionalOrderingType orderingType = std::nullopt,
      const gtsam::SemiringFactorGraph::Eliminate& function =
          gtsam::SemiringFactorGraph::Eliminate(
              gtsam::SemiringFactorGraph::EliminationTraitsType::DefaultEliminate),
      gtsam::SemiringFactorGraph::OptionalVariableIndex variableIndex = std::nullopt)
      const;
  std::shared_ptr<gtsam::SemiringBayesTree> eliminateMultifrontal(
      const gtsam::Ordering& ordering,
      const gtsam::SemiringFactorGraph::Eliminate& function =
          gtsam::SemiringFactorGraph::Eliminate(
              gtsam::SemiringFactorGraph::EliminationTraitsType::DefaultEliminate),
      gtsam::SemiringFactorGraph::OptionalVariableIndex variableIndex = std::nullopt)
      const;

  SemiringFactorGraph();
  SemiringFactorGraph(const gtsam::SemiringBayesNet& bayesNet);

  // Building the graph
  void push_back(const gtsam::SemiringFactor* factor);
  void push_back(const gtsam::SemiringFactorGraph& graph);
  void push_back(const gtsam::SemiringBayesNet& bayesNet);

  bool empty() const;
  size_t size() const;
  gtsam::KeySet keys() const;
  const gtsam::SemiringFactor* at(size_t i) const;

  void print(string s = "") const;
  bool equals(const gtsam::SemiringFactorGraph& fg, double tol = 1e-9) const;

  gtsam::SemiringFactor* product() const;
  double expectation(const gtsam::Ordering& ordering) const;
  double expectation() const;
};

}  // namespace gtsam
