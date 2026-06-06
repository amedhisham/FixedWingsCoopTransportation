#ifndef RTWCG_util_functions_h
#define RTWCG_util_functions_h

#if defined(STANDALONEFMU_UTIL_PREFIX)
  #define utilPaste(a,b)     a ## b
  #define utilPasteB(a,b)    utilPaste(a,b)
  #define utilFullName(name) utilPasteB(STANDALONEFMU_UTIL_PREFIX, name)
#else
  #define utilFullName(name) name
#endif

#define uriToLocal                                utilFullName(uriToLocal)
#define fmu_strncpy                               utilFullName(fmu_strncpy)
#define fmu_strndup                               utilFullName(fmu_strndup)
#define fmu_nestedDLLLocation                     utilFullName(fmu_nestedDLLLocation)
#define fmu_nestedResourceLocation                utilFullName(fmu_nestedResourceLocation)
#define raccelDeployLocation                      utilFullName(raccelDeployLocation)
#define fmu_resourceLocation                      utilFullName(fmu_resourceLocation)
#define fmu_deploymentDLLLocation                 utilFullName(fmu_deploymentDLLLocation)
#define fmu_deploymentResourceLocation            utilFullName(fmu_deploymentResourceLocation)
#define fmu_callback                              utilFullName(fmu_callback)
#define fmu_fmuLogging                            utilFullName(fmu_fmuLogging)
#define fmu_fmuVisible                            utilFullName(fmu_fmuVisible)
#define fmu_instanceName                          utilFullName(fmu_instanceName)
#define fmu_RestoreOutput                         utilFullName(fmu_RestoreOutput)
#define fmu_restoreSimScapeInitialState           utilFullName(fmu_restoreSimScapeInitialState)
#define fmu_syncFromBuffer                        utilFullName(fmu_syncFromBuffer)
#define fmu_syncToBuffer                          utilFullName(fmu_syncToBuffer)
#define fmu_parameterUpdated                      utilFullName(fmu_parameterUpdated)
#define fmu_clearParameterUpdated                 utilFullName(fmu_clearParameterUpdated)
#define rtOneStep                                 utilFullName(rtOneStep)
#define RT_MEMORY_ALLOCATION_ERROR                utilFullName(RT_MEMORY_ALLOCATION_ERROR)
#define _instance                                 utilFullName(_instance)
#define fmu_LogOutput                             utilFullName(fmu_LogOutput)
#define compareInputWithWorkingVec                utilFullName(compareInputWithWorkingVec)
#define updateVectorForFmu                        utilFullName(updateVectorForFmu)
#define updateVectorForBlk                        utilFullName(updateVectorForBlk)
#define fmu_loggerEnvironment                     utilFullName(fmu_loggerEnvironment)
#define loggerEnvironmentComponent                utilFullName(loggerEnvironmentComponent)
#define saveRapidSimulationMode                   utilFullName(saveRapidSimulationMode)

#endif