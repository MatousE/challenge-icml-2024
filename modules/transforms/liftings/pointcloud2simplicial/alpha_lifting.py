import torch
import torch_geometric
from scipy.spatial import Delaunay
from toponetx.classes import Simplex, SimplicialComplex

from modules.data.utils.utils import get_complex_connectivity
from modules.transforms.liftings.pointcloud2simplicial.base import (
    PointCloud2SimplicialLifting,
)


class SimplicialAlphaLifting(PointCloud2SimplicialLifting):
    def __init__(self, alpha: float, **kwargs):
        assert alpha > 0

        self.alpha = alpha
        super().__init__(**kwargs)

    def _get_lifted_topology(self, simplicial_complex: SimplicialComplex) -> dict:
        r"""Returns the lifted topology.
        Parameters
        ----------
        simplicial_complex : SimplicialComplex
            The simplicial complex.
        Returns
        -------
        dict
            The lifted topology.
        """
        lifted_topology = get_complex_connectivity(simplicial_complex, self.complex_dim)

        lifted_topology["x_0"] = torch.stack(
            list(simplicial_complex.get_simplex_attributes("features", 0).values())
        )

        return lifted_topology

    def circumradius(self, points):
        """Compute the circumradius of a simplex defined by the given points."""

        n = points.shape[0] - 1
        A = torch.cat([points[1:] - points[0], torch.ones(n, 1)], dim=1)
        b = torch.sum((points[1:] - points[0]) ** 2, dim=1) / 2
        x = torch.linalg.lstsq(A, b).solution

        # Compute the radius
        center = points[0] + x[:n]
        radius = torch.norm(points[0] - center)

        return radius.item()

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        points = data.pos

        delaunay = Delaunay(points.numpy())

        simplices = [
            Simplex(simplex)
            for simplex in delaunay.simplices
            if self.circumradius(points[simplex]) <= self.alpha
        ]

        SC = SimplicialComplex(simplices)
        self.complex_dim = SC.dim

        node_features = {i: data.x[i, :] for i in range(data.x.shape[0])}
        SC.set_simplex_attributes(node_features, name="features")

        return self._get_lifted_topology(SC)


# if __name__ == "__main__":
#     from modules.data.load.loaders import PointCloudLoader
#     from modules.utils.utils import describe_data
#
#     transform = AlphaLifting(alpha=1.0)
#
#     dataloader = PointCloudLoader(
#         {
#             "num_classes": 3,
#             "data_dir": "/Users/elphicm/PycharmProjects/challenge-icml-2024/modules/transforms/liftings/pointcloud2simplicial/",
#         }
#     )
#     data = dataloader.load()[0]
#
#     out = transform.lift_topology(data)
