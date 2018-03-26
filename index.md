# CMPT 489 AVX-512 Update 2

### Since you last saw us...

*What's happened with what we talked about last time*

Last time we showed off our updated `packh` and `packl` implementations. Since then, we've cleaned up our code and Prof. Cameron has added it to the main [icgrep svn repo](http://parabix.costar.sfu.ca/changeset/5931/icGREP/icgrep-devel).

Currently only supports field widths of 16.

Existing algorithm can be extended to field widths of 32 and 64 with the following:

````cpp
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

````cpp
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

````cpp
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

#### Bitblock advance with carry

##### Purpose

Long-stream addition (with carries) of 64 bit values.

##### Current algorithm

```cpp
std::pair<Value *, Value *> IDISA_AVX2_Builder::bitblock_add_with_carry(Value * e1, Value * e2, Value * carryin) {
    // using LONG_ADD
    Type * carryTy = carryin->getType();
    if (carryTy == mBitBlockType) {
        carryin = mvmd_extract(32, carryin, 0);
    }
    Value * carrygen = simd_and(e1, e2);
    Value * carryprop = simd_or(e1, e2);
    Value * digitsum = simd_add(64, e1, e2);
    Value * digitcarry = simd_or(carrygen, simd_and(carryprop, CreateNot(digitsum)));
    Value * carryMask = hsimd_signmask(64, digitcarry);
    Value * carryMask2 = CreateOr(CreateAdd(carryMask, carryMask), carryin);
    Value * bubble = simd_eq(64, digitsum, allOnes());
    Value * bubbleMask = hsimd_signmask(64, bubble);
    Value * incrementMask = CreateXor(CreateAdd(bubbleMask, carryMask2), bubbleMask);
    Value * increments = esimd_bitspread(64,incrementMask);
    Value * sum = simd_add(64, digitsum, increments);
    Value * carry_out = CreateLShr(incrementMask, mBitBlockWidth / 64);
    if (carryTy == mBitBlockType) {
        carry_out = bitCast(CreateZExt(carry_out, getIntNTy(mBitBlockWidth)));
    }
    return std::pair<Value *, Value *>{carry_out, bitCast(sum)};
}
```

##### Things to note

- Many nested logical operations
- Multiple layers of masks and bit manipulations

##### Problems

- No equivalent AVX-512 intrinsics have been found
- Field width is 64, but value size is not fixed

##### Research

Idea: Optimize performance for value sizes that are multiples of 512.

Drop-in replacements:

- ```_mm512_and_epi64```, ```_mm512_mask_or_epi64```, ```_mm512_xor_epi64```

Logical simplifications:

- ```_mm512_andnot_epi64``` can combine conjunction and negation
- Masked add intrinsics

##### Current work

In progress: implementing block-split algorithm and analyzing performance of intrinsic subtitutions.

### The LLVM Chronicles Pt.2: The Nine Circles of HeLLVM

*Rant about the BS LLVM nonsense by Oscar*

What started as quick detour to implement a single instruction version of `simd_srli` has turned into incredible journey through the depths of HeLLVM to uncover the mystical processes behind intrinsic implementation. So without further ado, our decent through the **Nine Circles of HeLLVM** starts as any other story does: with a naive curiosity straying a little deeper than it should've...

***
#### Limbo, The First Circle

We first start in **Limbo**. For a project investigating instruction-specific intrinsics in LLVM, life is always in Limbo. We face a state of constant unknown where we never know if a instruction really exists or if LLVM is just laughing at us.

***
#### Lust, The Second Circle

From there we decended to **Lust**. This is where the folly of man gets the best of us. One day, while browsing the Intel intrinsics of Limbo, we found this little gem:

```
__m512i _mm512_srl_epi16 (__m512i a, __m128i count)
Instruction: vpsrlw
CPUID Flags: AVX512BW
    
Shift packed 16-bit integers in a right by count while shifting in zeros,
and store the results in dst.
```
We quickly realized that this is *exactly* the same operation as the IDISA `simd_srli`! With this new discovery, we quickly became overtaken by Lust, the overwhelming urge to put our discovery to use.

***
#### Gluttony, The Third Circle

And that was where we fell into **Gluttony**. Normally, implementing a single-instruction override should be a piece of cake. We've already spent time figuring out the basic design needed, so it was just a matter of putting it all together. But just when we thought we'd succeeded, the glutton of LLVM reared its ugly head. *It just didn't work.*

```
icgrep ERROR: Cannot select: 0x37fd030: v32i16 = X86ISD::VSHLI 0x37f9120, Constant:i32<4>
```
***
#### Greed, The Fourth Circle

And before we could figure out why, we found ourselves already in **Greed**. With a little investigation, we discovered the glutton we'd faced: with so many features, there just isn't any documentation of anything that's happening in LLVM. The feature greed of the LLVM devs had condemned us, trapping us in a journey we'd never asked for. But we'd gotten this far, so we weren't going to stop.

***
#### Anger, The Fifth Circle

It was then that we entered **Anger**. We had discovered that for some reason, the enum which contains all the LLVM Intrinsics seemingly doesn't contain ours. Or at least, its existance is nondeterministic. What we'd discovered is that the Intrinsic enum is created using preprocessor macros which "simply" insert *over 6000* lines of Intrinsics into the body of LLVM's `Intrinsics.h`. They do this via dark magic and the file `Intrinsics.gen`.

***
#### Heresy, The Sixth Circle

Before we knew it, we found ourselves in **Heresy**. It seems the LLVM develoers, thinking themselves clever, had created one of the grossest ways to generate dynamic code bodies that had ever been concieved. The file `Intrinsics.gen` is *over 100,000* lines long and contains a number of pre-processor `ifdef`'s. By defining a specific preprocessor variable and then including this monstorous file, they load just the specific parts they want. This is how the 6k+ intrinsics are loaded into their enum.

| `Intrinsics.h`                                                                                                   | `Intrinsics.gen`                                                               |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `#define GET_INTRINSIC_ENUM_VALUES`<br>`#include "llvm/IR/Intrinsics.gen"`<br>`#undef GET_INTRINSIC_ENUM_VALUES` | `#ifdef GET_INTRINSIC_ENUM_VALUES`<br>`<6000 lines of Intrinsics>`<br>`#endif` |

***
#### Violence, The Seventh Circle

So now, we know how the Intrinsics are loaded and were even able to verify that ours does seem to exist there. So then, why doesn't it seem to exist? Before we could solve that problem, **Violence** reared its ugly head. Doing all this digging had inadvertantly opened *many* windows. Some were chrome tabs, some were new folders in VS Code. But together, they crashed our computer and set us back further than we'd like to admit.

```
  1  [||||||||||||||||||||||||||||||||                                                  46.6%]   
  2  [||||||||||||||||||||||||||||||||||||||||                                          51.2%]    
  3  [||||||||||||||||||||||||||||                                                      43.9%]   
  4  [||||||||||||||||||||||||||||||||||||                                              50.1%]
  Mem[||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||3.28G/3.43G]
  Swp[||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||7.89G/8.00G]
```
Having a computer with minimal RAM is a sad existence :(

***
#### Fraud, The Eighth Circle

After that little mishap, our journey continued. Figuring out what's wrong with our Intrinsic was the goal. But then, that we realized we'd entered **Fraud**. For all intents and purposes, our Intrinsic does exist in LLVM, so can we figure out what's broken? It exists in the Intrinsics list, our processor supports the instruction, there *seems* to be a pipeline for LLVM to compile the IR to asm. So then *why* does it not work? *Why* we asked, and an answer we were not given.

***
#### Treachery, The Ninth Circle

That was when our journey came to an end. We found ourselves at the bottom of HeLLVM: **Treachery**. We took a step back and decided to try some finagling with google search terms to see if maybe, just *maybe*, we could finally find someone else tackling a similar issue. What we found was not the salvation we were looking for, in fact, it was nearly the opposite. Turns out, every reference we could find for this issue, albeit with different intrinsics, has turned out to be an actual bug in the LLVM compiler.

| Error                                                                                     | Source                                                                                                                        |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `LLVM ERROR: Cannot select: intrinsic %llvm.x86.sse41.pblendvb`                           | [LLVM Toolchain Snapshot on Ubuntu Launchpad.](https://bugs.launchpad.net/ubuntu/+source/llvm-toolchain-snapshot/+bug/1389729) |
| `LLVM ERROR: Cannot select: 0x46ca590: v16f32 = X86ISD::FMAXC undef:v16f32, undef:v16f32` | [LLVM Bugzilla. Report for AVX-512 / KNL bug.](https://bugs.llvm.org/show_bug.cgi?id=27588)                                    |
| `LLVM ERROR: Cannot select: intrinsic %llvm.x86.avx512.rndscale.ss`                       | [LLVM Bugzilla. Report for seemingly broken unit test.](https://bugs.llvm.org/show_bug.cgi?id=20684)         <br> *Apparently the resolution to this bug was simply that they'd removed the Intrinsic. Not exactly a bode of confidence for us...*                  |



***
#### The End

As we climbed out of the deep hole we entered, we reflected on what we had learned. Even though we haven't conclusively shown that LLVM is at fault, something we're actively pursuing, there is strong evidence to suggest that this entire debacle is due to some small oversight in LLVM 3.8.0. While we can only hope that updating LLVM would indeed fix the issue, we have no guarantee. So for now, we'll try to nail down the issue and hopefully prevent anyone else from getting trapped in the deep, dark, endless hole that is **The Nine Circles of HeLLVM**.


### For the future

*What we're working on right now*

#### PExt and PDep

[Useful intel intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX_512&text=extract)

No work has been completed yet on PExt and PDep. Hope to have preliminary results next week.
