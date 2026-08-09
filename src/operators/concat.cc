#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    IT_ASSERT(!inputs.empty());
    Shape dims = inputs[0]->getDims();
    const auto rank = dims.size();
    for (size_t i = 1; i < inputs.size(); ++i) {
        const auto &input = inputs[i];
        IT_ASSERT(input->getRank() == rank);
        const auto inputDims = input->getDims();
        for (size_t axis = 0; axis < rank; ++axis) {
            if (axis != static_cast<size_t>(dim))
                IT_ASSERT(inputDims[axis] == dims[axis]);
        }
        dims[dim] += inputDims[dim];
    }

    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
