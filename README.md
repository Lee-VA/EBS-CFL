# EBS-CFL
 
We implemented the key algorithms and schemes from the paper.

### Key Algorithms:

- **VOMCA**: The core algorithm that encodes plaintext using pairwise orthogonal matrices. It ensures that without additional knowledge, decoding is impossible before all encoded data is aggregated. Additionally, the algorithm includes a verification function to ensure the encoding adheres to the required standards.
- **SRFC**: A Secure ReLU Function Calculation designed based on VOMCA. This allows for batch computation of the encoded ReLU function outputs without decoding. It is worth noting that the outputs remain in an encoded state and cannot be decoded before aggregation.
- **SKC**: A Secure Key Conversion algorithm designed based on VOMCA. It enables the transformation of an encoded key into another key without decoding and without leaking any key information.

### Schemes:

- **naive**: The basic scheme, corresponding to the encoding scheme described in the paper.
- **compress**: The compression scheme, corresponding to the compression method in the paper, which significantly improves computational and communication efficiency.


