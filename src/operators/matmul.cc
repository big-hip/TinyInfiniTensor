#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        IT_ASSERT(inputs.size() == 2);
        const auto a = inputs[0]->getDims();
        const auto b = inputs[1]->getDims();
        IT_ASSERT(a.size() >= 2 && b.size() >= 2);

        // The transpose attributes affect only the last two matrix axes.
        m = transA ? a[a.size() - 1] : a[a.size() - 2];
        const auto aK = transA ? a[a.size() - 2] : a[a.size() - 1];
        const auto bK = transB ? b[b.size() - 1] : b[b.size() - 2];
        n = transB ? b[b.size() - 2] : b[b.size() - 1];
        IT_ASSERT(aK == bK);
        k = aK;

        Shape aBatch(a.begin(), a.end() - 2);
        Shape bBatch(b.begin(), b.end() - 2);
        auto output = infer_broadcast(aBatch, bBatch);
        output.emplace_back(m);
        output.emplace_back(n);
        return {{output}};
    }

} // namespace infini
