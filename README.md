# Mercator2Azimuthal
This repository is for projecting panorama image in the way that human observes the reality world.

## How to use
There are two functions in the `method.py`, the `Mercator2Azimuthal` and the `Mercator2Azimuthal_with_mask`. The implementation in these two functions is the same.
The difference is 
- in `Mercator2Azimuthal_with_mask`, the mask of projection region is offered;
- the code is more friendly to people who would like to know the theory of the projection used in this repository;
- `Mercator2Azimuthal` is more efficient.

A demo is provided in the `main` function. If you meet any problem, never hesitate to contact me.

## Example

![](https://github.com/SalviaL/Mercator2Azimuthal/blob/main/image/result.png)
