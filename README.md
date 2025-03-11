# ST_traffic
# Paper Title: "A hybrid framework for spatio-temporal traffic flow prediction with multi-scale feature extraction"

## Authors
- **Lead Author**: [Ang Ji]

---

## Abstract
Efficient and accurate traffic flow prediction has become increasingly crucial with the advancement of intelligent transportation systems. This paper first extracts multi-scale features by depthwise separable convolution, which decomposes the convolution operation into independent spatial and temporal dimensions. This approach aims to reduce computational costs and effectively capture complex local spatio-temporal flow patterns in road networks. By adopting hierarchical processing, the model can learn dynamics across various scenarios, thereby enhancing its adaptability to different flow conditions. Then, we integrate a Transformer module into the model, leveraging its self-attention mechanism to capture the global relationships within traffic data. The Transformer can model long-range dependencies among road sections, enabling the understanding of the inter-dependencies of traffic flows across different regions. This capability is particularly beneficial in road networks with complex interaction effects. Experiments conducted on multiple real-world traffic datasets demonstrate that the proposed model outperforms traditional methods in both prediction accuracy and computational efficiency. The integration of depthwise separable convolution and Transformer enables the model to exhibit superior performance in traffic flow prediction, providing a sufficient tool for urban traffic management.
---
