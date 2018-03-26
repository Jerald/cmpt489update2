# CMPT 489 AVX-512 Update 2

### Since you last saw us...

*What's happened with what we talked about last time*

Last time we showed off our updated `packh` and `packl` implementations. Since then, we've cleaned up our code and Prof. Cameron has added it to the main [icgrep svn repo](http://parabix.costar.sfu.ca/changeset/5931/icGREP/icgrep-devel).

Currently only supports field widths of 16.

Existing algorithm can be extended to field widths of 32 and 64 with the following:

````
const unsigned int field_count_16 = 64;
Constant * Idxs_16[field_count_16];
for (unsigned int i = 0; i < field_count_16; i++) {
    Idxs_16[i] = getInt32(i);
}

const unsigned int field_count_32 = 32;
Constant * Idxs_32[field_count_32];
for (unsigned int i = 0; i < field_count_32; i++) {
    Idxs_32[i] = getInt32(i);
}

const unsigned int field_count_64 = 16;
Constant * Idxs_64[field_count_64];
for (unsigned int i = 0; i < field_count_64; i++) {
    Idxs_64[i] = getInt32(i);
}

switch (fw) {
    case 16:
        cvtfunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_pmov_wb_512);
        mask = getInt32(-1);
        shuffleMask = ConstantVector::get({Idxs_16, field_count_16});
        break;

    case 32:
        cvtfunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_pmov_dw_512);
        mask = getInt16(-1);
        shuffleMask = ConstantVector::get({Idxs_32, field_count_32});
        break;

    case 64:
        cvtfunc = Intrinsic::getDeclaration(getModule(), Intrinsic::x86_avx512_mask_pmov_qd_512);
        mask = getInt8(-1);
        shuffleMask = ConstantVector::get({Idxs_64, field_count_64});
        break;

    default:
        return IDISA_Builder::hsimd_packl(fw, a, b);
}
````

We are looking into a possible algorithm for field widths of 8 although it has
yet to produce results.

### The new stuff

*Start with the things from Cole*
#### Popcount

Default implementation:

````
if (LLVM_UNLIKELY(fw < 8)) {
    assert ("field width is less than 8" && false);
    llvm::report_fatal_error("Unsupported field width: popcount " + std::to_string(fw));
}
return CreatePopcount(fwCast(fw, a));
````

We had concerns that llvm may not be using the most efficient implementation.

##### Possible improvements

[Intel's PopCount Intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=pop)

[Solution we chose](https://en.wikipedia.org/wiki/Hamming_weight#Efficient_implementation)

````
llvm::Value * IDISA_AVX512BW_Builder::simd_popcount(unsigned fw, llvm::Value * a){
    if((fw == 64) && (mBitBlockWidth == 512)){
        Constant * m1Arr[8];
        llvm::Constant * m1;
        for (unsigned int i = 0; i < 8; i++) {
            m1Arr[i] = getInt64(0x5555555555555555);
        }
        m1 = ConstantVector::get({m1Arr, 8});

        Constant * m2Arr[8];
        llvm::Constant * m2;
        for (unsigned int i = 0; i < 8; i++) {
            m2Arr[i] = getInt64(0x3333333333333333);
        }
        m2 = ConstantVector::get({m2Arr, 8});

        Constant * m4Arr[8];
        llvm::Constant * m4;
        for (unsigned int i = 0; i < 8; i++) {
            m4Arr[i] = getInt64(0x0f0f0f0f0f0f0f0f);
        }
        m4 = ConstantVector::get({m4Arr, 8});

        Constant * h01Arr[8];
        llvm::Constant * h01;
        for (unsigned int i = 0; i < 8; i++) {
            h01Arr[i] = getInt64(0x0101010101010101);
        }
        h01 = ConstantVector::get({h01Arr, 8});

        a = simd_sub(fw, a, simd_and(simd_srli(fw, a, 1), m1));
        a = simd_add(fw, simd_and(a, m2), simd_and(simd_srli(fw, a, 2), m2));
        a = simd_and(simd_add(fw, a, simd_srli(fw, a, 4)), m4);
        return simd_srli(fw, simd_mult(fw, a, h01), 56);

    }
    return IDISA_Builder::simd_popcount(fw, a);
````

##### Results

Initial testing has shown roughly 2% gains in performance.

##### Further improvements

Algorithm can be modified for vectors of i32's, i16's and i8's.

*Then go to the things Avery has been working on*

### The LLVM Chronicles Pt.2: The Nine Circles of HeLLVM

*Rant about the BS LLVM nonsense by Oscar*

### For the future

*What we're working on right now*

#### PExt and PDep

[Useful intel intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX_512&text=extract)

No work has been completed yet on PExt and PDep. Hope to have preliminary results next week.
