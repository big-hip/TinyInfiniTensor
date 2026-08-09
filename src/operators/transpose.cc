#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
                transposePermute.emplace_back(rank - 1 - i);
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            vector<bool> seen(rank, false);
            for (auto axis : permute)
            {
                IT_ASSERT(axis >= 0 && axis < static_cast<int>(rank));
                IT_ASSERT(!seen[axis]);
                seen[axis] = true;
            }
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        auto output_dim = input_dim;
        for (size_t i = 0; i < transposePermute.size(); ++i)
            output_dim[i] = input_dim[transposePermute[i]];

        return {{output_dim}};
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
