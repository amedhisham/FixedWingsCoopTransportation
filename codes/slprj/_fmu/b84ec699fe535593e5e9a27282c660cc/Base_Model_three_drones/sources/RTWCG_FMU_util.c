#include "Base_Model_three_drones_macros.h"
/* Copyright 2021-2024 The MathWorks, Inc. */
/*
 * @file: RTWCG_FMU_util.c
 *  
 * @brief Definition of FMU path handling util function.
 *      
 */

#include "RTWCG_FMU_util.h"
#include <string.h>

#if FMU_CG_TARGET == FMUCG_MCC || FMU_CG_TARGET == FMUCG_GRT || FMU_CG_TARGET == FMUCG_NESTEDFMU || FMU_CG_TARGET == FMUCG_PROTECTED_MODEL
#ifdef _WIN64
    static const char* x64platform = "x86_64-windows";
    static const char* platform = "win64";
#elif defined(_WIN32)
    static const char* x64platform = "x86-windows";
    static const char* platform = "win32";
#elif defined(__APPLE__) && defined(__aarch64__)
    static const char* x64platform = "aarch64-darwin";
    static const char* platform = "aarch64-darwin";
#elif defined(__APPLE__) && defined(__x86_64__)
    static const char* x64platform = "x86_64-darwin";
    static const char* platform = "darwin64";
#elif defined(__linux__)
    static const char* x64platform = "x86_64-linux";
    static const char* platform = "linux64";
#elif defined(__QNXNTO__)
    static const char* x64platform = "x86_64-slrt";
    static const char* platform = "slrt_x64";
#else
    //# error Must specify OS.
#endif
#endif

#if FMU_CG_TARGET == FMUCG_MCC || FMU_CG_TARGET == FMUCG_GRT || FMU_CG_TARGET == FMUCG_NESTEDFMU || FMU_CG_TARGET == FMUCG_PROTECTED_MODEL
#if defined(_WIN64) || defined(_WIN32)
    static const char* libExt = ".dll";
#elif defined(__APPLE__)
    static const char* libExt = ".dylib";
#elif defined(__linux__) || defined(__QNXNTO__)
    static const char* libExt = ".so";
#else
    // # unknown platform cannot specify libExt for fmu binary
#endif
#endif

#if defined(_WIN64) || defined(_WIN32)
    static const char* separator = "\\";
#else
    static const char* separator = "/";
#endif

#if FMU_CG_TARGET == FMUCG_PROTECTED_MODEL || FMU_CG_TARGET == FMUCG_MCC || FMU_CG_TARGET == FMUCG_GRT
#ifdef _WIN32
#include <windows.h>     //GetModuleFileNameW
#elif __APPLE__          // maca64/maci64
#include <mach-o/dyld.h> // _NSGetExecutablePath
#else
#include <limits.h>
#include <unistd.h> //readlink
#endif

const char* raccelDeployLocation(void)
{
    static char exePath[1024];
#ifdef _WIN32
    wchar_t path[1024] = { 0 };
    GetModuleFileNameW(NULL, path, 1024);
    int sizeneeded = WideCharToMultiByte(CP_UTF8, 0, path, 1024, NULL, 0, NULL, NULL);
    WideCharToMultiByte(CP_UTF8, 0, path, 1024, &exePath[0], sizeneeded, NULL, NULL);
#elif __APPLE__
     uint32_t size = 1024;
    _NSGetExecutablePath(exePath, &size);
#else
    readlink("/proc/self/exe", exePath, 1024);
#endif
    return &exePath[0];
}
#endif


//protected model cg
#if FMU_CG_TARGET == FMUCG_PROTECTED_MODEL
#include "mex.h"
int saveRapidSimulationMode(int flag)
{
    static int rapidSimFlag;
    if(flag >= 0) {
        rapidSimFlag = flag;
    }
    return rapidSimFlag;
}

const char* fmu_resourceLocation(void) {
    static char dllPath[1024];
    const char* postFix = "/slprj/_fmu";
    memset(&dllPath[0], 0, sizeof(dllPath));
    if (saveRapidSimulationMode(-1) > 0) {
        const char* raccelPath = raccelDeployLocation();
        strcpy(dllPath, raccelPath);
#ifdef _WIN32
        char* pbase = strstr(&dllPath[0], "\\slprj\\raccel");
        *pbase = '\0';
#else
        char* pbase = strstr(&dllPath[0], "/slprj/raccel");
        *pbase = '\0';
#endif
        strcat(dllPath, postFix);
    }
    else {
        mxArray * output = NULL;
        mexCallMATLAB(1,&output, 0, NULL,"Simulink.ModelReference.ProtectedModel.getSimBuildDir");
        char* basePath = mxArrayToUTF8String(output);
        strcat(dllPath, basePath);
        strcat(dllPath, postFix);
        mxFree(basePath);
        mxDestroyArray(output);
    }
    return &dllPath[0];
}
#elif FMU_CG_TARGET == FMUCG_MCC || FMU_CG_TARGET == FMUCG_GRT || FMU_CG_TARGET == FMUCG_ERT
static void fmu2Logger(void* c,
                       const char* instanceName,
                       unsigned int status,
                       const char* category,
                       const char* message, ...) {
    (void)c;
    static const char* strStatus[] = {
        "fmi2OK", "fmi2Warning", "fmi2Discard", "fmi2Error", "fmi2Fatal", "fmi2Pending" };
    
    char translatedMsg[1024];
    char temp[1024];    
    int prefixLength = snprintf(translatedMsg, 1024, "Log from FMU %s: [category:%s, status:%s] ",
                            instanceName, strStatus[status], category); 
    va_list args;
    va_start (args, message);
    vsnprintf(temp, 1024, message, args);
    va_end(args);
    strncat(translatedMsg, temp, 1024-prefixLength - 1);

    printf("%s\n", translatedMsg);
}

static void FMU3_Logger(void* c,
                       unsigned int status,
                       const char* category,
                       const char* message) {
    (void)c;
    static const char* strStatus[] = {
        "fmi3OK", "fmi3Warning", "fmi3Discard", "fmi3Error", "fmi3Fatal"};

    char translatedMsg[1024];
    int prefixLength = snprintf(translatedMsg, 1024, "Log from FMU [category:%s, status:%s] ",
                             strStatus[status], category);
    strncat(translatedMsg, message, 1024-prefixLength - 1);

    printf("%s\n", translatedMsg);
}

const void* fmu_callback(int fmuVersion) {
    if (fmuVersion == 2) {
        typedef void      (*FMU2_CallbackLogger)        (void* componentEnvironment,
                                const char* instanceName,
                                unsigned int status,
                                const char* category,
                                const char* message,
                                ...);
        typedef void*     (*FMU2_CallbackAllocateMemory)(size_t nobj, size_t size);
        typedef void      (*FMU2_CallbackFreeMemory)    (void* obj);
        typedef void      (*FMU2_StepFinished)          (void* componentEnvironment,
                                                        unsigned int status);
        typedef struct { /* Local definition of FMU2_CallbackFunctions */
            FMU2_CallbackLogger         logger;
            FMU2_CallbackAllocateMemory allocateMemory;
            FMU2_CallbackFreeMemory     freeMemory;
            FMU2_StepFinished           stepFinished;
            void*   componentEnvironment;
        } FMU2_CallbackFunctions;
        static FMU2_CallbackFunctions callbacks = {fmu2Logger, calloc, free, NULL, NULL};
        callbacks.logger = fmu2Logger;
        callbacks.allocateMemory = calloc;
        callbacks.freeMemory = free;
        callbacks.stepFinished = NULL;
        return &callbacks;
    } else if (fmuVersion == 3) {
        typedef void  (*FMU3_LogMessageCallback) (void* instanceEnvironment,
                                                 unsigned int status,
                                                 const char* category,
                                                 const char* message);
        static FMU3_LogMessageCallback logMessage = FMU3_Logger;
        return &logMessage;
    } else{
        return NULL;
    }
}

#endif
    
#if FMU_CG_TARGET == FMUCG_NESTEDFMU || FMU_CG_TARGET == FMUCG_PROTECTED_MODEL
char* fmu_nestedDLLLocation(const char* uid, const char* model, const char* dllName, int x64Format) {
    static char dllPath[1024];
    memset(&dllPath[0], 0, sizeof(dllPath));
    const char* unziplocation = fmu_resourceLocation();
    strcat(dllPath, unziplocation);
    strcat(dllPath, separator);
    strcat(dllPath, uid);
    if (model != NULL) {
        strcat(dllPath, separator);
        strcat(dllPath, model);
    }
    strcat(dllPath, separator);
    strcat(dllPath, "binaries");
    strcat(dllPath, separator);
    if (x64Format == 1) {
        strcat(dllPath, x64platform);
    }
    else {
        strcat(dllPath, platform);
    }
    strcat(dllPath, separator);
    strcat(dllPath, dllName);
    strcat(dllPath, libExt);
    return dllPath;
}

char* fmu_nestedResourceLocation(const char* uid, const char* model, int root, int fmuVersion)
{
    static char nestedResourcesPath[1024];
    memset(&nestedResourcesPath[0], 0, sizeof(nestedResourcesPath));
    const char* resourceLoc = fmu_resourceLocation();
    if(fmuVersion <= 2) {
#ifdef _WIN32
        strcat(nestedResourcesPath, "file:///");
#else
        strcat(nestedResourcesPath, "file://");
#endif
    }
    strcat(nestedResourcesPath, resourceLoc);  
    strcat(nestedResourcesPath, separator);
    strcat(nestedResourcesPath, uid);
    if (model != NULL) {
        strcat(nestedResourcesPath, separator);
        strcat(nestedResourcesPath, model);
    }
    if (root == 0) {
        strcat(nestedResourcesPath, separator);
        strcat(nestedResourcesPath, "resources");
    }
    return nestedResourcesPath;
}
#endif

#if FMU_CG_TARGET == FMUCG_MCC || FMU_CG_TARGET == FMUCG_GRT
const char* fmu_resourceLocation(void) {
    return raccelDeployLocation();
}

char* fmu_deploymentDLLLocation(const char* deploy, const char* uid, const char* model, const char* dllName, int x64Format)
{
    static char dllPath[1024];
    memset(&dllPath[0], 0, sizeof(dllPath));
    const char* unziplocation = raccelDeployLocation();
    strcat(dllPath, unziplocation);
    char* pbase = strrchr(dllPath, *separator);
    *pbase = '\0';
    if (deploy != NULL) {
    	strcat(dllPath, separator);
        strcat(dllPath, deploy);
    }
    strcat(dllPath, separator);
    strcat(dllPath, "_fmu");
    strcat(dllPath, separator);
    strcat(dllPath, uid);
    if (model != NULL) {
        strcat(dllPath, separator);
        strcat(dllPath, model);
    }
    strcat(dllPath, separator);
    strcat(dllPath, "binaries");
    strcat(dllPath, separator);
    if (x64Format == 1) {
        strcat(dllPath, x64platform);
    }
    else {
        strcat(dllPath, platform);
    }
    strcat(dllPath, separator);
    strcat(dllPath, dllName);
    strcat(dllPath, libExt);
    return dllPath;
}

char* fmu_deploymentResourceLocation(const char* deploy, const char* uid, const char* model, int root, int fmuVersion)
{
    static char raccelResourcesPath[1024];
    memset(&raccelResourcesPath[0], 0, sizeof(raccelResourcesPath));
    const char* unziplocation = raccelDeployLocation();
    if(fmuVersion <= 2) {
#ifdef _WIN32
    strcat(raccelResourcesPath, "file:///");
#else
    strcat(raccelResourcesPath, "file://");
#endif
    }
    strcat(raccelResourcesPath, unziplocation);
    char* pbase = strrchr(raccelResourcesPath, *separator);
    *pbase = '\0';
    if (deploy != NULL) {
    	strcat(raccelResourcesPath, separator);
        strcat(raccelResourcesPath, deploy);
    }
    strcat(raccelResourcesPath, separator);
    strcat(raccelResourcesPath, "_fmu");
    strcat(raccelResourcesPath, separator);
    strcat(raccelResourcesPath, uid);
    if (model != NULL) {
        strcat(raccelResourcesPath, separator);
        strcat(raccelResourcesPath, model);
    }
    if (root == 0) {
        strcat(raccelResourcesPath, separator);
        strcat(raccelResourcesPath, "resources");
    }
    return raccelResourcesPath;
}
#endif

static int hex2Dec(char c) {
    if(c<= '9' && c >= '0') {
        return c - '0';
    }
    else if(c <= 'F' && c>= 'A') {
        return c - 'A' + 10;
    }
    else if(c <= 'f' && c>= 'a') {
        return c - 'a' + 10;
    }
    return -1;
}

static char URIEncoding(const char enc[3]) {
    int dig1 = hex2Dec(enc[1]);
    int dig2 = hex2Dec(enc[2]);
    //handle 00 - FF
    if( dig1 < 0 || dig1 > 15  || dig2 < 0 || dig2 > 15) {
        return'\0';
    }
    return dig1 * 16 + dig2;
}

static void copyToString(const char* c, char* s) {
    const char* begin = c;
    while (*c != '\0') {
        if (*c == '%' && (*(c + 1) !='\0') && (*(c + 2) != '\0')) {
            *s = URIEncoding(c);
            c += 3;
        }
#ifdef _WIN32
        else if(*c == '/') {
            *s = *separator;
            c++;
        }
#endif
        else {
            *s = *c;
            c++;
        }
        if (*s == '\0') {
            return;
        }
        s++;
    }
    s--;
    c--;
    while (*s == *separator && c != begin) {
        *s = '\0';
        s--;
        c--;
    }
}

/*uriToLocal is capable to handle paths with percent encoding:
 *      file://localhost/path/to/fmu
 *      file:/path/to/fmu
 *      file:///path/to/fmu
 */

char* uriToLocal(const char *uri) {

	const char *scheme1 = "file:///";
	const char *scheme2 = "file:/";
	const char* scheme3 = "http";
	const char* scheme4 = "ftp";
	const char* localhost = "/localhost/";
	char* path = (char*)calloc(1024, sizeof(char));

	if (strncmp(uri, scheme1, strlen(scheme1)) == 0) {
        const char *c = &uri[strlen(scheme1)];        
#if defined(__APPLE__) || defined(__linux__) || defined(__QNXNTO__)
        c--;
#endif
		copyToString(c, path);
	} else if (strncmp(uri, scheme2, strlen(scheme2)) == 0) {
		const char *c = &uri[strlen(scheme2)];
        if(strncmp(c, localhost, strlen(localhost)) == 0) {
			c = &uri[strlen(scheme2) + strlen(localhost)];
		}
#if defined(__APPLE__) || defined(__linux__) || defined(__QNXNTO__)
        c--;
#endif
		copyToString(c, path);
	} else if(strncmp(uri, scheme3, strlen(scheme3)) == 0) {
		//do not handle
	} else if(strncmp(uri, scheme4, strlen(scheme4)) == 0) {
		//do not handle
	} else {
		strcpy(path, uri);
	}

	return path;
}

void fmu_strncpy(char* dest, void* src, int size) {
    strncpy(dest, (char*)src, size);
    dest[size] = '\x00';
}

char* fmu_strndup(const char* src, int size) {
    size_t srclen = strlen(src);
    size_t copylen = srclen < size? srclen : size;
    char *dest= (char*)malloc(copylen+1);
    if (dest != NULL) {
        fmu_strncpy(dest, (void*)src, copylen);
        }
    return dest;
}

/*================================================================*

 * Functions to support LinIdx for Row-major and Col-major arrays *

 *================================================================*/

static void incrementIdxVectorColumnMajor( unsigned int *idxVector,  const unsigned int *inDim, const int  nDim) {
    int  i;
    if (!idxVector || !inDim)
        return;
    i = 0;
    while (i < nDim) {
        idxVector[i]++;
        if (idxVector[i] >= inDim[i] ) { idxVector[i++] = 0; }
        else { break; }
    }
}

static void incrementIdxVectorRowMajor( unsigned int *idxVector,  const unsigned int *inDim, const int  nDim) {
    int  i;
    if (!idxVector || !inDim)
        return;
    i = nDim-1;
    while (i>= 0) {
        idxVector[i]++;
        if (idxVector[i] >= inDim[i] ) { idxVector[i--] = 0; }
        else { break; }
    }

}


static int getColMajorLinIdxFromIdxVec(const unsigned int *idxVector, const unsigned int *inDim, const int nDim){
    int linIdx, stride, i;
    if (!idxVector || !inDim)
        return -1;

    linIdx=0;
    stride=1;
    for (i=0; i<nDim; i++){
         linIdx += stride*idxVector[i];
         stride *= inDim[i];
    }
    return linIdx;

}

static int getRowMajorLinIdxFromIdxVec(const unsigned int *idxVector, const unsigned int *inDim, const int nDim){
    int linIdx, stride, i;
    if (!idxVector || !inDim)
        return -1;

    int numOfTotalElements = 1;
    for (i=0; i<nDim; i++)
         numOfTotalElements *= inDim[i];

    linIdx=0;
    stride = numOfTotalElements;
    for (i=0; i<nDim; i++)
    {
        stride /= inDim[i];
        linIdx += stride*idxVector[i];
    }
    return linIdx;
}

void compareInputWithWorkingVec(void *dstWorkingVec, const void* srcBlkInput, const unsigned int* dimList, const size_t nVal, const size_t dstWorkingVecDTSize, const size_t srcBlkInputDTSize, int needConversion, int *is_diff_value) {
    // Compare the input data with the working vector and set the is_diff_value
    // flag accordingly.
    // Use this method when the datatypes for dstWorkingVec and srcBlkInput
    // are compatible. If the datatypes are not compatible then using this
    // method may produce unintended result.
    unsigned int idxVector[64]; // using 64 as the max number of dimensions.
    int nDim, linIdx, j;
    nDim = dimList[0]; // Skipping, max dims check
    const unsigned int *inDim = &dimList[1]; // Skipping, max dims check

    if(!dstWorkingVec || !srcBlkInput || !dimList) {
        return;
    }

    // initialize index vector
    for (j = 0; j < nDim; j++){
        idxVector[j] = 0;
    }

    // Handle direct comparison case first, when 
    // datatypes are size-compatible and no conversion is needed
    if(needConversion == 0 && (dstWorkingVecDTSize == srcBlkInputDTSize)) {
        *is_diff_value = *is_diff_value | (memcmp(dstWorkingVec, srcBlkInput, nVal*dstWorkingVecDTSize) != 0);
        return;
    }

    // find the minimum of sizes to compare
    size_t minSize = (srcBlkInputDTSize < dstWorkingVecDTSize) ? srcBlkInputDTSize : dstWorkingVecDTSize;

    for (j = 0; j < nVal; j++) {
        linIdx = getColMajorLinIdxFromIdxVec(idxVector, inDim, nDim);
        // using memcmp is safe because the datatypes are compatible.
        *is_diff_value = *is_diff_value | (memcmp(((char*)dstWorkingVec + (j*dstWorkingVecDTSize)), 
                                                 ((char*)srcBlkInput + (linIdx*srcBlkInputDTSize)), 
                                                 minSize) != 0);
        incrementIdxVectorRowMajor(idxVector, inDim, nDim);
    }
}

void updateVectorForFmu(void *dstWorkingVec, const void* srcBlkInput, const unsigned int* dimList, const size_t nVal, const size_t dstDTSize, const size_t srcDTSize, int needConversion) {
    // Copies data from source block input to destination working vector while handling:
    // - Different data type sizes
    // - Array layout conversions (column-major to row-major)
    // - Multi-dimensional array support (up to 64 dimensions)
    // Use this method when the datatypes for dstWorkingVec and srcBlkInput
    // are compatible. If the datatypes are not compatible then using this
    // method may produce unintended result due to overflow.
    unsigned int idxVector[64]; // using 64 as the max number of dimensions.
    int nDim, linIdx, j;
    nDim = dimList[0]; // Skipping, max dims check
    const unsigned int *inDim = &dimList[1]; // Skipping, max dims check
    
    if(!dstWorkingVec || !srcBlkInput || !dimList) {
        return;
    }

    // initialize index vector
    for (j = 0; j < nDim; j++){
        idxVector[j] = 0;
    }

    // Handle direct copy case first, when 
    // datatypes are size-compatible and no conversion is needed
    if(needConversion == 0 && (dstDTSize == srcDTSize)) {
        memcpy(dstWorkingVec, srcBlkInput, nVal*srcDTSize);
        return;
    }

    // find the minimum of srcDTSize and dstDTSize
    size_t minSize = (srcDTSize < dstDTSize) ? srcDTSize : dstDTSize;

    for (j = 0; j < nVal; j++) {
        linIdx = getColMajorLinIdxFromIdxVec(idxVector, inDim, nDim);
        memcpy(((char*)dstWorkingVec + (j*dstDTSize)), 
               ((char*)srcBlkInput + (linIdx*srcDTSize)), 
               minSize);
        incrementIdxVectorRowMajor(idxVector, inDim, nDim);
    }
}

void updateVectorForBlk(void *dstBlkOutput, const void* srcWorkingVector, const unsigned int* dimList, const size_t nVal, const size_t dstDTSize, const size_t srcDTSize, int needConversion) {
    // Copies data from destination block output to working vector while handling:
    // - Different data type sizes
    // - Array layout conversions (column-major to row-major)
    // - Multi-dimensional array support (up to 64 dimensions)
    // Use this method when the datatypes for dstBlkOutput and srcWorkingVector
    // are compatible. If the datatypes are not compatible then using this
    // method may produce unintended result due to overflow.
    unsigned int idxVector[64]; // using 64 as the max number of dimensions.
    int nDim, linIdx, j;
    nDim = dimList[0]; // Skipping, max dims check
    const unsigned int *inDim = &dimList[1]; // Skipping, max dims check

    if(!dstBlkOutput || !srcWorkingVector || !dimList) {
        return;
    }
    // initialize index vector
    for (j = 0; j < nDim; j++){
        idxVector[j] = 0;
    }

    // Handle direct copy case first
    if(needConversion == 0 && (srcDTSize == dstDTSize)) {
        memcpy(dstBlkOutput, srcWorkingVector, nVal*srcDTSize);
        return;
    }

    // find the minimum of srcDTSize and dstDTSize
    size_t minSize = (srcDTSize < dstDTSize) ? srcDTSize : dstDTSize;

    // Handle both enum and non-enum cases with same loop structure
    // Original comment: find the minimum of of datasize and dstDTSize
    for (j = 0; j < nVal; j++) {
        linIdx = getRowMajorLinIdxFromIdxVec(idxVector, inDim, nDim);
        memcpy(((char*)dstBlkOutput + (j*dstDTSize)), 
               ((char*)srcWorkingVector + (linIdx*srcDTSize)), 
               minSize);
        incrementIdxVectorColumnMajor(idxVector, inDim, nDim);
    }
}

void* fmu_loggerEnvironment(int enableLogging, const char* fmuname)
{
    static struct loggerEnvironmentComponent comp = {0, NULL};
    if (comp.instanceName == NULL) {
        comp.isLoggingOn = enableLogging;
        comp.instanceName = fmuname;
    }
    return &comp;
}