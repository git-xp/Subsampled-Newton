# Subsampled-Newton

Codes for subsampled Newton methods for solving ridge logistic regression.

## About

Consider minimizing a sum of convex functions. Subsampled-Newton methods subsample the functions to calcuate the approximated Hessian. Using non-uniform sampling schemes, we can show the sampling size can be independent of number of functions. In typical ERM problems where n >> d, sub-sampled Newton methods can speed up a lot.

## Usage
- Matlab users: See <code> subsampled_newton.m </code> for main functions. And see <code>demo_comparison.m</code> for usage.

- Python users: See files in the folder <code>python</code>.

## Reference

Peng Xu, Jiyan Yang, Farbod Roosta-Khorasani, Christopher Ré, and Michael W. Mahoney, [Sub-sampled Newton Methods with Non-uniform Sampling](https://papers.nips.cc/paper/6037-sub-sampled-newton-methods-with-non-uniform-sampling.pdf), NIPS 2016.

