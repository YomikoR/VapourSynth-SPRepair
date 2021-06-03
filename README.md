# VapourSynth-SPRepair
Sub-pixel repair for certain modes

## Note
Currently only modes 0-4, 11-14 are implemented.

## Usage

```python
sprep.spRepair(clip, repairclip, mode, pixel=1.0)
```
The first three parameters are the same as `rgvs.Repair`.

The last parameter `pixel` is a floating-point number specifying the pixel shift of the `repairclip` to achieve sub-pixel operation.

Internally, `repairclip` is taken shifts by `pixel` into 8 neighboring directions, which take the place of grid neighborhoods used by `rgvs.Repair`.
Setting `pixel` to `1.0` will make them identical, except borders.

## Acknowledgment
The sorting network used in the program is from the appendix of

Vinod K Valsalam and Risto Miikkulainen, *Using Symmetry and Evolutionary Search to Minimize Sorting Networks*, Journal of Machine Learning Research 14 (2013) 303-331.
