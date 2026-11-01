// The slice of the LLVM-C API airc uses, implemented over our own bitcode reader (llvm_lite.c) so airc can
// run where there is no libLLVM - the Metal plugin. Same names and semantics as
// llvm-c/Core.h for the functions declared here; anything else is not implemented.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef int LLVMBool;
typedef struct ll_value *LLVMValueRef;
typedef struct ll_block *LLVMBasicBlockRef;
typedef struct ll_module *LLVMModuleRef;
typedef struct ll_membuf *LLVMMemoryBufferRef;
typedef struct ll_type *LLVMTypeRef;

typedef enum {
    LLVMVoidTypeKind,
    LLVMHalfTypeKind,
    LLVMBFloatTypeKind,
    LLVMFloatTypeKind,
    LLVMDoubleTypeKind,
    LLVMFP128TypeKind,
    LLVMLabelTypeKind,
    LLVMIntegerTypeKind,
    LLVMFunctionTypeKind,
    LLVMStructTypeKind,
    LLVMArrayTypeKind,
    LLVMPointerTypeKind,
    LLVMVectorTypeKind,
    LLVMMetadataTypeKind,
    LLVMTokenTypeKind,
} LLVMTypeKind;

typedef enum {
    LLVMUnknownOp = 0,
    LLVMRet,
    LLVMUncondBr,
    LLVMCondBr,
    LLVMSwitch,
    LLVMUnreachable,
    LLVMFNeg,
    LLVMAdd,
    LLVMFAdd,
    LLVMSub,
    LLVMFSub,
    LLVMMul,
    LLVMFMul,
    LLVMUDiv,
    LLVMSDiv,
    LLVMFDiv,
    LLVMURem,
    LLVMSRem,
    LLVMFRem,
    LLVMShl,
    LLVMLShr,
    LLVMAShr,
    LLVMAnd,
    LLVMOr,
    LLVMXor,
    LLVMAlloca,
    LLVMLoad,
    LLVMStore,
    LLVMGetElementPtr,
    LLVMTrunc,
    LLVMZExt,
    LLVMSExt,
    LLVMFPToUI,
    LLVMFPToSI,
    LLVMUIToFP,
    LLVMSIToFP,
    LLVMFPTrunc,
    LLVMFPExt,
    LLVMPtrToInt,
    LLVMIntToPtr,
    LLVMBitCast,
    LLVMAddrSpaceCast,
    LLVMICmp,
    LLVMFCmp,
    LLVMPHI,
    LLVMCall,
    LLVMSelect,
    LLVMExtractElement,
    LLVMInsertElement,
    LLVMShuffleVector,
    LLVMExtractValue,
    LLVMInsertValue,
    LLVMFreeze,
    LLVMAtomicRMW,
    LLVMAtomicCmpXchg,
    LLVMFence,
} LLVMOpcode;

// numerically identical to the bitcode / CmpInst::Predicate encoding
typedef enum {
    LLVMIntEQ = 32,
    LLVMIntNE,
    LLVMIntUGT,
    LLVMIntUGE,
    LLVMIntULT,
    LLVMIntULE,
    LLVMIntSGT,
    LLVMIntSGE,
    LLVMIntSLT,
    LLVMIntSLE,
} LLVMIntPredicate;

typedef enum {
    LLVMRealPredicateFalse,
    LLVMRealOEQ,
    LLVMRealOGT,
    LLVMRealOGE,
    LLVMRealOLT,
    LLVMRealOLE,
    LLVMRealONE,
    LLVMRealORD,
    LLVMRealUNO,
    LLVMRealUEQ,
    LLVMRealUGT,
    LLVMRealUGE,
    LLVMRealULT,
    LLVMRealULE,
    LLVMRealUNE,
    LLVMRealPredicateTrue,
} LLVMRealPredicate;

LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRange(const char *data, size_t len, const char *name,
                                                          LLVMBool requires_null);
LLVMBool LLVMParseBitcode2(LLVMMemoryBufferRef buf, LLVMModuleRef *out);
void LLVMDisposeModule(LLVMModuleRef m);
void LLVMDisposeMemoryBuffer(LLVMMemoryBufferRef buf);

unsigned LLVMGetNamedMetadataNumOperands(LLVMModuleRef m, const char *name);
void LLVMGetNamedMetadataOperands(LLVMModuleRef m, const char *name, LLVMValueRef *out);
unsigned LLVMGetMDNodeNumOperands(LLVMValueRef v);
void LLVMGetMDNodeOperands(LLVMValueRef v, LLVMValueRef *out);
const char *LLVMGetMDString(LLVMValueRef v, unsigned *len);

LLVMTypeRef LLVMTypeOf(LLVMValueRef v);
LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef t);
unsigned LLVMGetIntTypeWidth(LLVMTypeRef t);
LLVMTypeRef LLVMGetGEPSourceElementType(LLVMValueRef gep);
unsigned LLVMGetVectorSize(LLVMTypeRef t);
unsigned LLVMGetAlignment(LLVMValueRef loadOrStore);  // bytes, 0 when the bitcode gives none
LLVMTypeRef LLVMGetElementType(LLVMTypeRef t);
unsigned LLVMGetPointerAddressSpace(LLVMTypeRef t);
uint64_t LLVMGetArrayLength2(LLVMTypeRef t);
unsigned LLVMCountStructElementTypes(LLVMTypeRef t);
unsigned LLVMGetNumIndices(LLVMValueRef v);
const unsigned *LLVMGetIndices(LLVMValueRef v);
LLVMTypeRef LLVMStructGetTypeAtIndex(LLVMTypeRef t, unsigned i);
LLVMValueRef LLVMIsAGlobalVariable(LLVMValueRef v);
LLVMTypeRef LLVMGlobalGetValueType(LLVMValueRef g);

LLVMValueRef LLVMIsAConstantExpr(LLVMValueRef v);  // only getelementptr expressions are read
LLVMOpcode LLVMGetConstOpcode(LLVMValueRef v);
LLVMBool LLVMIsUndef(LLVMValueRef v);
LLVMBool LLVMIsPoison(LLVMValueRef v);
LLVMValueRef LLVMGetInitializer(LLVMValueRef g);
LLVMTypeRef LLVMGetAllocatedType(LLVMValueRef alloca);
LLVMTypeRef LLVMInt32Type(void);
LLVMTypeRef LLVMInt8Type(void);
LLVMValueRef LLVMGetAggregateElement(LLVMValueRef c, unsigned i);  // a constant vector's element i
unsigned LLVMGetNumMaskElements(LLVMValueRef shuffle);
int LLVMGetMaskValue(LLVMValueRef shuffle, unsigned i);  // LLVMGetUndefMaskElem() for an undefined lane
int LLVMGetUndefMaskElem(void);

const char *LLVMGetValueName(LLVMValueRef v);
LLVMValueRef LLVMIsAFunction(LLVMValueRef v);
LLVMValueRef LLVMIsAInstruction(LLVMValueRef v);
LLVMValueRef LLVMIsAICmpInst(LLVMValueRef v);
LLVMValueRef LLVMIsAConstantInt(LLVMValueRef v);
LLVMValueRef LLVMIsAConstantFP(LLVMValueRef v);
unsigned long long LLVMConstIntGetZExtValue(LLVMValueRef v);
double LLVMConstRealGetDouble(LLVMValueRef v, LLVMBool *loses_info);

unsigned LLVMCountParams(LLVMValueRef fn);
LLVMValueRef LLVMGetParam(LLVMValueRef fn, unsigned i);
LLVMBasicBlockRef LLVMGetFirstBasicBlock(LLVMValueRef fn);
LLVMValueRef LLVMGetFirstFunction(LLVMModuleRef m);
LLVMValueRef LLVMGetNextFunction(LLVMValueRef fn);
LLVMBasicBlockRef LLVMGetNextBasicBlock(LLVMBasicBlockRef bb);
LLVMValueRef LLVMGetFirstInstruction(LLVMBasicBlockRef bb);
LLVMValueRef LLVMGetNextInstruction(LLVMValueRef v);
LLVMValueRef LLVMGetBasicBlockTerminator(LLVMBasicBlockRef bb);

LLVMOpcode LLVMGetInstructionOpcode(LLVMValueRef v);
LLVMBool LLVMCanContract(LLVMValueRef v);  // a floating-point instruction with the contract or fast flag
int LLVMGetNumOperands(LLVMValueRef v);
LLVMValueRef LLVMGetOperand(LLVMValueRef v, unsigned i);
LLVMIntPredicate LLVMGetICmpPredicate(LLVMValueRef v);
LLVMRealPredicate LLVMGetFCmpPredicate(LLVMValueRef v);
LLVMValueRef LLVMGetCalledValue(LLVMValueRef call);
// function attributes: only noinline is known
typedef struct ll_value *LLVMAttributeRef;
typedef unsigned LLVMAttributeIndex;
enum { LLVMAttributeFunctionIndex = ~0u };
unsigned LLVMGetEnumAttributeKindForName(const char *name, size_t len);
LLVMAttributeRef LLVMGetEnumAttributeAtIndex(LLVMValueRef fn, LLVMAttributeIndex idx, unsigned kind);
LLVMValueRef LLVMGetCondition(LLVMValueRef br);
unsigned LLVMGetNumSuccessors(LLVMValueRef term);
LLVMBasicBlockRef LLVMGetSuccessor(LLVMValueRef term, unsigned i);
LLVMValueRef LLVMGetSwitchCaseValue(LLVMValueRef sw, unsigned i);  // i = the successor, from 1
unsigned LLVMCountIncoming(LLVMValueRef phi);
LLVMValueRef LLVMGetIncomingValue(LLVMValueRef phi, unsigned i);
LLVMBasicBlockRef LLVMGetIncomingBlock(LLVMValueRef phi, unsigned i);
