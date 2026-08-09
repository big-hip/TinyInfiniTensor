#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        IT_ASSERT(topo_sort());

        // Rebuild all reverse edges after changing operator inputs or removing
        // operators. Doing this in two passes ensures consumers are known when
        // predecessor/successor edges are reconstructed.
        auto rebuildConnections = [this]()
        {
            for (const auto &tensor : tensors)
            {
                tensor->targets.clear();
                tensor->source.reset();
            }
            for (const auto &op : ops)
            {
                op->predecessors.clear();
                op->successors.clear();
            }

            for (const auto &op : ops)
            {
                for (const auto &output : op->outputs)
                    if (output)
                        output->setSource(op);
                for (const auto &input : op->inputs)
                    if (input)
                        input->addTarget(op);
            }

            for (const auto &op : ops)
            {
                for (const auto &input : op->inputs)
                {
                    if (input)
                    {
                        if (auto pred = input->getSource())
                        {
                            pred->addSuccessors(op);
                            op->addPredecessors(pred);
                        }
                    }
                }
            }
            sorted = false;
        };

        auto areInversePermutes = [](const vector<int> &first,
                                     const vector<int> &second)
        {
            if (first.size() != second.size())
                return false;
            for (size_t i = 0; i < first.size(); ++i)
                if (first[second[i]] != static_cast<int>(i))
                    return false;
            return true;
        };

        auto swapsLastTwoAxes = [](const TransposeObj &transpose)
        {
            const auto permute = transpose.getPermute();
            if (permute.size() < 2)
                return false;
            const auto rank = permute.size();
            for (size_t i = 0; i + 2 < rank; ++i)
                if (permute[i] != static_cast<int>(i))
                    return false;
            return permute[rank - 2] == static_cast<int>(rank - 1) &&
                   permute[rank - 1] == static_cast<int>(rank - 2);
        };

        bool changed;
        do
        {
            changed = false;
            rebuildConnections();

            // Remove adjacent inverse transpose operators. The output of the
            // second transpose is replaced at all of its consumers by the
            // input of the first transpose.
            for (const auto &first : ops)
            {
                auto firstTranspose = as<TransposeObj>(first);
                if (!firstTranspose)
                    continue;
                auto middle = firstTranspose->getOutput();
                auto middleTargets = middle->getTargets();
                if (middleTargets.size() != 1)
                    continue;
                auto second = middleTargets[0];
                auto secondTranspose = as<TransposeObj>(second);
                if (!secondTranspose || secondTranspose->getInputs(0) != middle ||
                    !areInversePermutes(firstTranspose->getPermute(),
                                        secondTranspose->getPermute()))
                    continue;

                auto original = firstTranspose->getInputs(0);
                auto result = secondTranspose->getOutput();
                for (const auto &user : result->getTargets())
                    user->replaceInput(result, original);

                ops.erase(std::remove(ops.begin(), ops.end(), first), ops.end());
                ops.erase(std::remove(ops.begin(), ops.end(), second), ops.end());
                changed = true;
                break;
            }
            if (changed)
                continue;

            // Fold a transpose that only swaps the last two axes into MatMul.
            for (const auto &op : ops)
            {
                auto matmul = as<MatmulObj>(op);
                if (!matmul)
                    continue;
                for (size_t inputIndex = 0; inputIndex < 2; ++inputIndex)
                {
                    auto transposed = matmul->getInputs(inputIndex);
                    auto sourceOp = transposed->getSource();
                    auto transpose = sourceOp ? as<TransposeObj>(sourceOp) : nullptr;
                    if (!transpose || transpose->getOutput() != transposed ||
                        transposed->getTargets().size() != 1 ||
                        !swapsLastTwoAxes(*transpose))
                        continue;

                    matmul->replaceInput(transposed, transpose->getInputs(0));
                    if (inputIndex == 0)
                        matmul->setTransA(!matmul->getTransA());
                    else
                        matmul->setTransB(!matmul->getTransB());

                    auto shape = matmul->inferShape(matmul->getInputs());
                    IT_ASSERT(shape.has_value() && shape->size() == 1);
                    IT_ASSERT(shape->at(0) == matmul->getOutput()->getDims());

                    ops.erase(std::remove(ops.begin(), ops.end(), sourceOp),
                              ops.end());
                    changed = true;
                    break;
                }
                if (changed)
                    break;
            }
        } while (changed);

        rebuildConnections();
        tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                     [](const Tensor &tensor)
                                     {
                                         return !tensor->getSource() &&
                                                tensor->getTargets().empty();
                                     }),
                      tensors.end());
        sorted = false;
        IT_ASSERT(topo_sort());
        IT_ASSERT(checkValid());
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // Count uses in execution order. A tensor remains live until its last
        // consumer has finished, allowing the allocator to reuse its storage.
        map<Tensor, size_t> remainingUses;
        for (const auto &op : ops)
            for (const auto &input : op->getInputs())
                ++remainingUses[input];

        map<Tensor, size_t> offsets;
        for (const auto &op : ops)
        {
            for (const auto &input : op->getInputs())
            {
                if (offsets.find(input) == offsets.end())
                    offsets.emplace(input, allocator.alloc(input->getBytes()));
            }
            for (const auto &output : op->getOutputs())
            {
                IT_ASSERT(offsets.find(output) == offsets.end());
                offsets.emplace(output, allocator.alloc(output->getBytes()));
            }

            for (const auto &input : op->getInputs())
            {
                auto uses = --remainingUses[input];
                if (uses == 0)
                    allocator.free(offsets.at(input), input->getBytes());
            }
        }

        auto base = static_cast<char *>(allocator.getPtr());
        for (const auto &[tensor, offset] : offsets)
            tensor->setDataBlob(make_ref<BlobObj>(runtime, base + offset));

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
