/*************************************************************
 * 
 *    kpp_integrate_cuda_prototype.cu
 *    Prototype file for kpp CUDA kernel
 *
 *    Copyright 2016 The Cyprus Institute 
 *
 *    Developers: Michail Alvanos - m.alvanos@cyi.ac.cy
 *                Giannis Ashiotis
 *                Theodoros Christoudias - christoudias@cyi.ac.cy
 *
 ********************************************************************/

#include <stdio.h>
#include <unistd.h>
#include "cuda.h"

#define NSPEC 142
#define NVAR 139
#define NFIX 3
#define NREACT 310
#define LU_NONZERO 1486
#define NBSIZE 523
#define BLOCKSIZE 64

//#define MAX_VL_GLO 12288 /* elements that will pass in each call */

#define REDUCTION_SIZE_1 64
#define REDUCTION_SIZE_2 32

#define R_gas 8.3144621
#define N_A 6.02214129e+23
#define atm2Pa 101325.0

#define ip_O2 0
#define ip_O3P 1
#define ip_O1D 2
#define ip_H2O2 3
#define ip_NO2 4
#define ip_NO2O 5
#define ip_NOO2 6
#define ip_N2O5 7
#define ip_HNO3 8
#define ip_HNO4 9
#define ip_PAN 10
#define ip_HONO 11
#define ip_CH3OOH 12
#define ip_COH2 13
#define ip_CHOH 14
#define ip_CH3CO3H 15
#define ip_CH3CHO 16
#define ip_CH3COCH3 17
#define ip_MGLYOX 18
#define ip_HOCl 19
#define ip_OClO 20
#define ip_Cl2O2 21
#define ip_ClNO3 22
#define ip_ClNO2 23
#define ip_Cl2 24
#define ip_BrO 25
#define ip_HOBr 26
#define ip_BrCl 27
#define ip_BrNO3 28
#define ip_BrNO2 29
#define ip_Br2 30
#define ip_CCl4 31
#define ip_CH3Cl 32
#define ip_CH3CCl3 33
#define ip_CFCl3 34
#define ip_CF2Cl2 35
#define ip_CH3Br 36
#define ip_CF2ClBr 37
#define ip_CF3Br 38
#define ip_CH3I 39
#define ip_C3H7I 40
#define ip_CH2ClI 41
#define ip_CH2I2 42
#define ip_IO 43
#define ip_HOI 44
#define ip_I2 45
#define ip_ICl 46
#define ip_IBr 47
#define ip_INO2 48
#define ip_INO3 49
#define ip_SO2 50
#define ip_SO3 51
#define ip_OCS 52
#define ip_CS2 53
#define ip_H2O 54
#define ip_N2O 55
#define ip_NO 56
#define ip_CO2 57
#define ip_HCl 58
#define ip_CHCl2Br 59
#define ip_CHClBr2 60
#define ip_CH2ClBr 61
#define ip_CH2Br2 62
#define ip_CHBr3 63
#define ip_SF6 64
#define ip_NO3NOO 65
#define ip_ClONO2 66
#define ip_MACR 67
#define ip_MVK 68
#define ip_GLYOX 69
#define ip_HOCH2CHO 70
#define ip_CH4 71
#define ip_O2_b1b2 72
#define ip_O2_b1 73
#define ip_O2_b2 74
#define ip_O3PO1D 75
#define ip_O3Pp 76
#define ip_H2O1D 77
#define ip_N2 78
#define ip_N2_b1 79
#define ip_N2_b2 80
#define ip_N2_b3 81
#define ip_NN2D 82
#define ip_NOp 83
#define ip_Op_em 84
#define ip_O2p_em 85
#define ip_Op_O_em 86
#define ip_N2p_em 87
#define ip_Np_N_em 88
#define ip_Np_N2D_em 89
#define ip_N_N2D_em 90
#define ip_Op_em_b 91
#define ip_se_O2_b1 92
#define ip_se_O2_b2 93
#define ip_se_N2_b1 94
#define ip_se_N2_b2 95
#define ip_se_N2_b3 96
#define ip_se_N2_b4 97
#define ip_se_Op_em 98
#define ip_O2_aurq 99
#define ip_N2_aurq 100
#define ip_H2SO4 101
#define ip_C3O2 102
#define ip_CH3NO3 103
#define ip_CH3O2NO2 104
#define ip_CH3ONO 105
#define ip_CH3O2 106
#define ip_HCOOH 107
#define ip_HO2NO2 108
#define ip_OHNO3 109
#define ip_qqqdummy 110
#define ip_CH3OCl 111
#define ip_MEO2NO2 112
#define ip_CHF2Cl 113
#define ip_F113 114
#define ip_C2H5NO3 115
#define ip_NOA 116
#define ip_MEKNO3 117
#define ip_BENZAL 118
#define ip_HOPh3Me2NO2 119
#define ip_HOC6H4NO2 120
#define ip_CH3CHO2VINY 121
#define ip_CH3COCO2H 122
#define ip_IPRCHO2HCO 123
#define ip_C2H5CHO2HCO 124
#define ip_C2H5CHO2ENOL 125
#define ip_C3H7CHO2HCO 126
#define ip_C3H7CHO2VINY 127
#define ip_PeDIONE24 128
#define ip_PINAL2HCO 129
#define ip_PINAL2ENOL 130
#define ip_CF2ClCFCl2 131
#define ip_CH3CFCl2 132
#define ip_CF3CF2Cl 133
#define ip_CF2ClCF2Cl 134
#define ip_CHCl3 135
#define ip_CH2Cl2 136
#define ip_HO2 137
#define ip_ClO 138

#define ind_BrNO2 0
#define ind_CF2ClBr 1
#define ind_CF3Br 2
#define ind_CH3I 3
#define ind_O3s 4
#define ind_CF2ClBr_c 5
#define ind_CF3Br_c 6
#define ind_LCARBON 7
#define ind_LFLUORINE 8
#define ind_LCHLORINE 9
#define ind_CH3SO3H 10
#define ind_H2SO4 11
#define ind_NO3m_cs 12
#define ind_Hp_cs 13
#define ind_Dummy 14
#define ind_CFCl3_c 15
#define ind_CF2Cl2_c 16
#define ind_N2O_c 17
#define ind_CH3CCl3_c 18
#define ind_LO3s 19
#define ind_LossHO2 20
#define ind_LossO1D 21
#define ind_LossO3 22
#define ind_LossO3Br 23
#define ind_LossO3Cl 24
#define ind_LossO3H 25
#define ind_LossO3N 26
#define ind_LossO3O 27
#define ind_LossO3R 28
#define ind_LossOH 29
#define ind_ProdHO2 30
#define ind_ProdLBr 31
#define ind_ProdLCl 32
#define ind_ProdMeO2 33
#define ind_ProdO3 34
#define ind_ProdRO2 35
#define ind_ProdSBr 36
#define ind_ProdSCl 37
#define ind_BIACET 38
#define ind_Cl2O2 39
#define ind_NC4H10 40
#define ind_CCl4 41
#define ind_CF2Cl2 42
#define ind_CFCl3 43
#define ind_CH2Br2 44
#define ind_CHBr3 45
#define ind_CH3SO3 46
#define ind_NH3 47
#define ind_C2H6 48
#define ind_C3H8 49
#define ind_ClNO2 50
#define ind_OClO 51
#define ind_CH2ClBr 52
#define ind_CH3Br 53
#define ind_CHCl2Br 54
#define ind_CHClBr2 55
#define ind_SO2 56
#define ind_CH3CCl3 57
#define ind_NACA 58
#define ind_N 59
#define ind_N2O 60
#define ind_NH2OH 61
#define ind_IC3H7NO3 62
#define ind_CH3CO3H 63
#define ind_MPAN 64
#define ind_DMSO 65
#define ind_ISOOH 66
#define ind_LHOC3H6OOH 67
#define ind_LMEKOOH 68
#define ind_IC3H7OOH 69
#define ind_NHOH 70
#define ind_C2H5OOH 71
#define ind_HYPERACET 72
#define ind_HNO4 73
#define ind_CH3CO2H 74
#define ind_CH3Cl 75
#define ind_HONO 76
#define ind_PAN 77
#define ind_HCOOH 78
#define ind_LC4H9OOH 79
#define ind_Cl2 80
#define ind_CH3SO2 81
#define ind_MVKOOH 82
#define ind_N2O5 83
#define ind_NH2O 84
#define ind_MEK 85
#define ind_CH3COCH3 86
#define ind_HNO 87
#define ind_H2O2 88
#define ind_CH3OH 89
#define ind_BrCl 90
#define ind_ISON 91
#define ind_NH2 92
#define ind_IC3H7O2 93
#define ind_CH3COCH2O2 94
#define ind_CO 95
#define ind_MGLYOX 96
#define ind_H2 97
#define ind_CH4 98
#define ind_LMEKO2 99
#define ind_Br2 100
#define ind_HNO3 101
#define ind_LC4H9O2 102
#define ind_C2H4 103
#define ind_CH3OOH 104
#define ind_BrNO3 105
#define ind_C5H8 106
#define ind_C3H6 107
#define ind_ACETOL 108
#define ind_ISO2 109
#define ind_MVK 110
#define ind_LC4H9NO3 111
#define ind_HOCl 112
#define ind_MVKO2 113
#define ind_DMS 114
#define ind_LHOC3H6O2 115
#define ind_ClNO3 116
#define ind_C2H5O2 117
#define ind_HOBr 118
#define ind_CH3CHO 119
#define ind_O1D 120
#define ind_CH3CO3 121
#define ind_H 122
#define ind_HBr 123
#define ind_O3 124
#define ind_CH3O2 125
#define ind_OH 126
#define ind_Cl 127
#define ind_H2O 128
#define ind_Br 129
#define ind_HCHO 130
#define ind_O3P 131
#define ind_BrO 132
#define ind_NO 133
#define ind_ClO 134
#define ind_NO2 135
#define ind_NO3 136
#define ind_HO2 137
#define ind_HCl 138
#define ind_O2 139
#define ind_N2 140
#define ind_CO2 141
#define ind_H2OH2O -1
#define ind_N2D -1
#define ind_LNITROGEN -1
#define ind_CH2OO -1
#define ind_CH2OOA -1
#define ind_CH3 -1
#define ind_CH3O -1
#define ind_HOCH2O2 -1
#define ind_HOCH2OH -1
#define ind_HOCH2OOH -1
#define ind_CH3NO3 -1
#define ind_CH3O2NO2 -1
#define ind_CH3ONO -1
#define ind_CN -1
#define ind_HCN -1
#define ind_HOCH2O2NO2 -1
#define ind_NCO -1
#define ind_C2H2 -1
#define ind_C2H5OH -1
#define ind_CH2CHOH -1
#define ind_CH2CO -1
#define ind_CH3CHOHO2 -1
#define ind_CH3CHOHOOH -1
#define ind_CH3CO -1
#define ind_ETHGLY -1
#define ind_GLYOX -1
#define ind_HCOCH2O2 -1
#define ind_HCOCO -1
#define ind_HCOCO2H -1
#define ind_HCOCO3 -1
#define ind_HCOCO3H -1
#define ind_HOCH2CH2O -1
#define ind_HOCH2CH2O2 -1
#define ind_HOCH2CHO -1
#define ind_HOCH2CO -1
#define ind_HOCH2CO2H -1
#define ind_HOCH2CO3 -1
#define ind_HOCH2CO3H -1
#define ind_HOCHCHO -1
#define ind_HOOCH2CHO -1
#define ind_HOOCH2CO2H -1
#define ind_HOOCH2CO3 -1
#define ind_HOOCH2CO3H -1
#define ind_HYETHO2H -1
#define ind_C2H5NO3 -1
#define ind_C2H5O2NO2 -1
#define ind_CH3CN -1
#define ind_ETHOHNO3 -1
#define ind_NCCH2O2 -1
#define ind_NO3CH2CHO -1
#define ind_NO3CH2CO3 -1
#define ind_NO3CH2PAN -1
#define ind_PHAN -1
#define ind_ALCOCH2OOH -1
#define ind_C2H5CHO -1
#define ind_C2H5CO2H -1
#define ind_C2H5CO3 -1
#define ind_C2H5CO3H -1
#define ind_C33CO -1
#define ind_CH3CHCO -1
#define ind_CH3COCO2H -1
#define ind_CH3COCO3 -1
#define ind_CH3COCO3H -1
#define ind_CHOCOCH2O2 -1
#define ind_HCOCH2CHO -1
#define ind_HCOCH2CO2H -1
#define ind_HCOCH2CO3 -1
#define ind_HCOCH2CO3H -1
#define ind_HCOCOCH2OOH -1
#define ind_HOC2H4CO2H -1
#define ind_HOC2H4CO3 -1
#define ind_HOC2H4CO3H -1
#define ind_HOCH2COCH2O2 -1
#define ind_HOCH2COCH2OOH -1
#define ind_HOCH2COCHO -1
#define ind_HYPROPO2 -1
#define ind_HYPROPO2H -1
#define ind_IPROPOL -1
#define ind_NC3H7O2 -1
#define ind_NC3H7OOH -1
#define ind_NPROPOL -1
#define ind_PROPENOL -1
#define ind_C32OH13CO -1
#define ind_C3DIALO2 -1
#define ind_C3DIALOOH -1
#define ind_HCOCOHCO3 -1
#define ind_HCOCOHCO3H -1
#define ind_METACETHO -1
#define ind_C3PAN1 -1
#define ind_C3PAN2 -1
#define ind_CH3COCH2O2NO2 -1
#define ind_NC3H7NO3 -1
#define ind_NOA -1
#define ind_PPN -1
#define ind_PR2O2HNO3 -1
#define ind_PRONO3BO2 -1
#define ind_PROPOLNO3 -1
#define ind_HCOCOHPAN -1
#define ind_BIACETO2 -1
#define ind_BIACETOH -1
#define ind_BIACETOOH -1
#define ind_BUT1ENE -1
#define ind_BUT2OLO -1
#define ind_BUT2OLO2 -1
#define ind_BUT2OLOOH -1
#define ind_BUTENOL -1
#define ind_C312COCO3 -1
#define ind_C312COCO3H -1
#define ind_C3H7CHO -1
#define ind_C413COOOH -1
#define ind_C44O2 -1
#define ind_C44OOH -1
#define ind_C4CODIAL -1
#define ind_CBUT2ENE -1
#define ind_CH3COCHCO -1
#define ind_CH3COCHO2CHO -1
#define ind_CH3COCOCO2H -1
#define ind_CH3COOHCHCHO -1
#define ind_CHOC3COO2 -1
#define ind_CO23C3CHO -1
#define ind_CO2C3CHO -1
#define ind_CO2H3CHO -1
#define ind_CO2H3CO2H -1
#define ind_CO2H3CO3 -1
#define ind_CO2H3CO3H -1
#define ind_EZCH3CO2CHCHO -1
#define ind_EZCHOCCH3CHO2 -1
#define ind_HCOCCH3CHOOH -1
#define ind_HCOCCH3CO -1
#define ind_HCOCO2CH3CHO -1
#define ind_HMAC -1
#define ind_HO12CO3C4 -1
#define ind_HVMK -1
#define ind_IBUTALOH -1
#define ind_IBUTDIAL -1
#define ind_IBUTOLBO2 -1
#define ind_IBUTOLBOOH -1
#define ind_IC4H10 -1
#define ind_IC4H9O2 -1
#define ind_IC4H9OOH -1
#define ind_IPRCHO -1
#define ind_IPRCO3 -1
#define ind_IPRHOCO2H -1
#define ind_IPRHOCO3 -1
#define ind_IPRHOCO3H -1
#define ind_MACO2 -1
#define ind_MACO2H -1
#define ind_MACO3 -1
#define ind_MACO3H -1
#define ind_MACR -1
#define ind_MACRO -1
#define ind_MACRO2 -1
#define ind_MACROH -1
#define ind_MACROOH -1
#define ind_MBOOO -1
#define ind_MEPROPENE -1
#define ind_MPROPENOL -1
#define ind_PERIBUACID -1
#define ind_TBUT2ENE -1
#define ind_TC4H9O2 -1
#define ind_TC4H9OOH -1
#define ind_BZFUCO -1
#define ind_BZFUO2 -1
#define ind_BZFUONE -1
#define ind_BZFUOOH -1
#define ind_CO14O3CHO -1
#define ind_CO14O3CO2H -1
#define ind_CO2C4DIAL -1
#define ind_EPXC4DIAL -1
#define ind_EPXDLCO2H -1
#define ind_EPXDLCO3 -1
#define ind_EPXDLCO3H -1
#define ind_HOCOC4DIAL -1
#define ind_MALANHY -1
#define ind_MALANHYO2 -1
#define ind_MALANHYOOH -1
#define ind_MALDALCO2H -1
#define ind_MALDALCO3H -1
#define ind_MALDIAL -1
#define ind_MALDIALCO3 -1
#define ind_MALDIALO2 -1
#define ind_MALDIALOOH -1
#define ind_MALNHYOHCO -1
#define ind_MECOACEOOH -1
#define ind_MECOACETO2 -1
#define ind_BUT2OLNO3 -1
#define ind_C312COPAN -1
#define ind_C4PAN5 -1
#define ind_IBUTOLBNO3 -1
#define ind_IC4H9NO3 -1
#define ind_MACRN -1
#define ind_MVKNO3 -1
#define ind_PIPN -1
#define ind_TC4H9NO3 -1
#define ind_EPXDLPAN -1
#define ind_MALDIALPAN -1
#define ind_NBZFUO2 -1
#define ind_NBZFUONE -1
#define ind_NBZFUOOH -1
#define ind_NC4DCO2H -1
#define ind_LBUT1ENO2 -1
#define ind_LBUT1ENOOH -1
#define ind_LHMVKABO2 -1
#define ind_LHMVKABOOH -1
#define ind_LBUT1ENNO3 -1
#define ind_LMEKNO3 -1
#define ind_C1ODC2O2C4OD -1
#define ind_C1ODC2O2C4OOH -1
#define ind_C1ODC2OOHC4OD -1
#define ind_C1ODC3O2C4OOH -1
#define ind_C1OOHC2O2C4OD -1
#define ind_C1OOHC2OOHC4OD -1
#define ind_C1OOHC3O2C4OD -1
#define ind_C4MDIAL -1
#define ind_C511O2 -1
#define ind_C511OOH -1
#define ind_C512O2 -1
#define ind_C512OOH -1
#define ind_C513CO -1
#define ind_C513O2 -1
#define ind_C513OOH -1
#define ind_C514O2 -1
#define ind_C514OOH -1
#define ind_C59O2 -1
#define ind_C59OOH -1
#define ind_CHOC3COCO3 -1
#define ind_CHOC3COOOH -1
#define ind_CO13C4CHO -1
#define ind_CO23C4CHO -1
#define ind_CO23C4CO3 -1
#define ind_CO23C4CO3H -1
#define ind_DB1O -1
#define ind_DB1O2 -1
#define ind_DB1OOH -1
#define ind_DB2O2 -1
#define ind_DB2OOH -1
#define ind_HCOC5 -1
#define ind_ISOPAB -1
#define ind_ISOPAOH -1
#define ind_ISOPBO2 -1
#define ind_ISOPBOH -1
#define ind_ISOPBOOH -1
#define ind_ISOPCD -1
#define ind_ISOPDO2 -1
#define ind_ISOPDOH -1
#define ind_ISOPDOOH -1
#define ind_MBO -1
#define ind_MBOACO -1
#define ind_MBOCOCO -1
#define ind_ME3FURAN -1
#define ind_ZCO3C23DBCOD -1
#define ind_ZCODC23DBCOOH -1
#define ind_ACCOMECHO -1
#define ind_ACCOMECO3 -1
#define ind_ACCOMECO3H -1
#define ind_C24O3CCO2H -1
#define ind_C4CO2DBCO3 -1
#define ind_C4CO2DCO3H -1
#define ind_C5134CO2OH -1
#define ind_C54CO -1
#define ind_C5CO14O2 -1
#define ind_C5CO14OH -1
#define ind_C5CO14OOH -1
#define ind_C5DIALCO -1
#define ind_C5DIALO2 -1
#define ind_C5DIALOOH -1
#define ind_C5DICARB -1
#define ind_C5DICARBO2 -1
#define ind_C5DICAROOH -1
#define ind_MC3ODBCO2H -1
#define ind_MMALANHY -1
#define ind_MMALANHYO2 -1
#define ind_MMALNHYOOH -1
#define ind_TLFUO2 -1
#define ind_TLFUONE -1
#define ind_TLFUOOH -1
#define ind_C4MCONO3OH -1
#define ind_C514NO3 -1
#define ind_C5PAN9 -1
#define ind_CHOC3COPAN -1
#define ind_DB1NO3 -1
#define ind_ISOPBDNO3O2 -1
#define ind_ISOPBNO3 -1
#define ind_ISOPDNO3 -1
#define ind_NC4CHO -1
#define ind_NC4OHCO3 -1
#define ind_NC4OHCO3H -1
#define ind_NC4OHCPAN -1
#define ind_NISOPO2 -1
#define ind_NISOPOOH -1
#define ind_NMBOBCO -1
#define ind_ZCPANC23DBCOD -1
#define ind_ACCOMEPAN -1
#define ind_C4CO2DBPAN -1
#define ind_C5COO2NO2 -1
#define ind_NC4MDCO2H -1
#define ind_NTLFUO2 -1
#define ind_NTLFUOOH -1
#define ind_LC578O2 -1
#define ind_LC578OOH -1
#define ind_LDISOPACO -1
#define ind_LDISOPACO2 -1
#define ind_LHC4ACCHO -1
#define ind_LHC4ACCO2H -1
#define ind_LHC4ACCO3 -1
#define ind_LHC4ACCO3H -1
#define ind_LIEPOX -1
#define ind_LISOPACO -1
#define ind_LISOPACO2 -1
#define ind_LISOPACOOH -1
#define ind_LISOPEFO -1
#define ind_LISOPEFO2 -1
#define ind_LMBOABO2 -1
#define ind_LMBOABOOH -1
#define ind_LME3FURANO2 -1
#define ind_LZCO3HC23DBCOD -1
#define ind_LC5PAN1719 -1
#define ind_LISOPACNO3 -1
#define ind_LISOPACNO3O2 -1
#define ind_LMBOABNO3 -1
#define ind_LNISO3 -1
#define ind_LNISOOH -1
#define ind_LNMBOABO2 -1
#define ind_LNMBOABOOH -1
#define ind_C614CO -1
#define ind_C614O2 -1
#define ind_C614OOH -1
#define ind_CO235C5CHO -1
#define ind_CO235C6O2 -1
#define ind_CO235C6OOH -1
#define ind_BENZENE -1
#define ind_BZBIPERO2 -1
#define ind_BZBIPEROOH -1
#define ind_BZEMUCCO -1
#define ind_BZEMUCCO2H -1
#define ind_BZEMUCCO3 -1
#define ind_BZEMUCCO3H -1
#define ind_BZEMUCO2 -1
#define ind_BZEMUCOOH -1
#define ind_BZEPOXMUC -1
#define ind_BZOBIPEROH -1
#define ind_C5CO2DBCO3 -1
#define ind_C5CO2DCO3H -1
#define ind_C5CO2OHCO3 -1
#define ind_C5COOHCO3H -1
#define ind_C6125CO -1
#define ind_C615CO2O2 -1
#define ind_C615CO2OOH -1
#define ind_C6CO4DB -1
#define ind_C6H5O -1
#define ind_C6H5O2 -1
#define ind_C6H5OOH -1
#define ind_CATEC1O -1
#define ind_CATEC1O2 -1
#define ind_CATEC1OOH -1
#define ind_CATECHOL -1
#define ind_CPDKETENE -1
#define ind_PBZQCO -1
#define ind_PBZQO2 -1
#define ind_PBZQONE -1
#define ind_PBZQOOH -1
#define ind_PHENO2 -1
#define ind_PHENOL -1
#define ind_PHENOOH -1
#define ind_C614NO3 -1
#define ind_BZBIPERNO3 -1
#define ind_BZEMUCNO3 -1
#define ind_BZEMUCPAN -1
#define ind_C5CO2DBPAN -1
#define ind_C5CO2OHPAN -1
#define ind_DNPHEN -1
#define ind_DNPHENO2 -1
#define ind_DNPHENOOH -1
#define ind_HOC6H4NO2 -1
#define ind_NBZQO2 -1
#define ind_NBZQOOH -1
#define ind_NCATECHOL -1
#define ind_NCATECO2 -1
#define ind_NCATECOOH -1
#define ind_NCPDKETENE -1
#define ind_NDNPHENO2 -1
#define ind_NDNPHENOOH -1
#define ind_NNCATECO2 -1
#define ind_NNCATECOOH -1
#define ind_NPHEN1O -1
#define ind_NPHEN1O2 -1
#define ind_NPHEN1OOH -1
#define ind_NPHENO2 -1
#define ind_NPHENOOH -1
#define ind_C235C6CO3H -1
#define ind_C716O2 -1
#define ind_C716OOH -1
#define ind_C721O2 -1
#define ind_C721OOH -1
#define ind_C722O2 -1
#define ind_C722OOH -1
#define ind_CO235C6CHO -1
#define ind_CO235C6CO3 -1
#define ind_MCPDKETENE -1
#define ind_ROO6R3O -1
#define ind_ROO6R3O2 -1
#define ind_ROO6R5O2 -1
#define ind_BENZAL -1
#define ind_C6CO2OHCO3 -1
#define ind_C6COOHCO3H -1
#define ind_C6H5CH2O2 -1
#define ind_C6H5CH2OOH -1
#define ind_C6H5CO3 -1
#define ind_C6H5CO3H -1
#define ind_C7CO4DB -1
#define ind_CRESO2 -1
#define ind_CRESOL -1
#define ind_CRESOOH -1
#define ind_MCATEC1O -1
#define ind_MCATEC1O2 -1
#define ind_MCATEC1OOH -1
#define ind_MCATECHOL -1
#define ind_OXYL1O2 -1
#define ind_OXYL1OOH -1
#define ind_PHCOOH -1
#define ind_PTLQCO -1
#define ind_PTLQO2 -1
#define ind_PTLQONE -1
#define ind_PTLQOOH -1
#define ind_TLBIPERO2 -1
#define ind_TLBIPEROOH -1
#define ind_TLEMUCCO -1
#define ind_TLEMUCCO2H -1
#define ind_TLEMUCCO3 -1
#define ind_TLEMUCCO3H -1
#define ind_TLEMUCO2 -1
#define ind_TLEMUCOOH -1
#define ind_TLEPOXMUC -1
#define ind_TLOBIPEROH -1
#define ind_TOL1O -1
#define ind_TOLUENE -1
#define ind_C7PAN3 -1
#define ind_C6CO2OHPAN -1
#define ind_C6H5CH2NO3 -1
#define ind_DNCRES -1
#define ind_DNCRESO2 -1
#define ind_DNCRESOOH -1
#define ind_MNCATECH -1
#define ind_MNCATECO2 -1
#define ind_MNCATECOOH -1
#define ind_MNCPDKETENE -1
#define ind_MNNCATCOOH -1
#define ind_MNNCATECO2 -1
#define ind_NCRES1O -1
#define ind_NCRES1O2 -1
#define ind_NCRES1OOH -1
#define ind_NCRESO2 -1
#define ind_NCRESOOH -1
#define ind_NDNCRESO2 -1
#define ind_NDNCRESOOH -1
#define ind_NPTLQO2 -1
#define ind_NPTLQOOH -1
#define ind_PBZN -1
#define ind_TLBIPERNO3 -1
#define ind_TLEMUCNO3 -1
#define ind_TLEMUCPAN -1
#define ind_TOL1OHNO2 -1
#define ind_C721CHO -1
#define ind_C721CO3 -1
#define ind_C721CO3H -1
#define ind_C810O2 -1
#define ind_C810OOH -1
#define ind_C811O2 -1
#define ind_C812O2 -1
#define ind_C812OOH -1
#define ind_C813O2 -1
#define ind_C813OOH -1
#define ind_C85O2 -1
#define ind_C85OOH -1
#define ind_C86O2 -1
#define ind_C86OOH -1
#define ind_C89O2 -1
#define ind_C89OOH -1
#define ind_C8BC -1
#define ind_C8BCCO -1
#define ind_C8BCO2 -1
#define ind_C8BCOOH -1
#define ind_NORPINIC -1
#define ind_EBENZ -1
#define ind_LXYL -1
#define ind_STYRENE -1
#define ind_STYRENO2 -1
#define ind_STYRENOOH -1
#define ind_C721PAN -1
#define ind_C810NO3 -1
#define ind_C89NO3 -1
#define ind_C8BCNO3 -1
#define ind_NSTYRENO2 -1
#define ind_NSTYRENOOH -1
#define ind_C811CO3 -1
#define ind_C811CO3H -1
#define ind_C85CO3 -1
#define ind_C85CO3H -1
#define ind_C89CO2H -1
#define ind_C89CO3 -1
#define ind_C89CO3H -1
#define ind_C96O2 -1
#define ind_C96OOH -1
#define ind_C97O2 -1
#define ind_C97OOH -1
#define ind_C98O2 -1
#define ind_C98OOH -1
#define ind_NOPINDCO -1
#define ind_NOPINDO2 -1
#define ind_NOPINDOOH -1
#define ind_NOPINONE -1
#define ind_NOPINOO -1
#define ind_NORPINAL -1
#define ind_NORPINENOL -1
#define ind_PINIC -1
#define ind_RO6R3P -1
#define ind_C811PAN -1
#define ind_C89PAN -1
#define ind_C96NO3 -1
#define ind_C9PAN2 -1
#define ind_LTMB -1
#define ind_APINAOO -1
#define ind_APINBOO -1
#define ind_APINENE -1
#define ind_BPINAO2 -1
#define ind_BPINAOOH -1
#define ind_BPINENE -1
#define ind_C106O2 -1
#define ind_C106OOH -1
#define ind_C109CO -1
#define ind_C109O2 -1
#define ind_C109OOH -1
#define ind_C96CO3 -1
#define ind_CAMPHENE -1
#define ind_CARENE -1
#define ind_MENTHEN6ONE -1
#define ind_OH2MENTHEN6ONE -1
#define ind_OHMENTHEN6ONEO2 -1
#define ind_PERPINONIC -1
#define ind_PINAL -1
#define ind_PINALO2 -1
#define ind_PINALOOH -1
#define ind_PINENOL -1
#define ind_PINONIC -1
#define ind_RO6R1O2 -1
#define ind_RO6R3O2 -1
#define ind_RO6R3OOH -1
#define ind_ROO6R1O2 -1
#define ind_SABINENE -1
#define ind_BPINANO3 -1
#define ind_C106NO3 -1
#define ind_C10PAN2 -1
#define ind_PINALNO3 -1
#define ind_RO6R1NO3 -1
#define ind_RO6R3NO3 -1
#define ind_ROO6R1NO3 -1
#define ind_LAPINABNO3 -1
#define ind_LAPINABO2 -1
#define ind_LAPINABOOH -1
#define ind_LNAPINABO2 -1
#define ind_LNAPINABOOH -1
#define ind_LNBPINABO2 -1
#define ind_LNBPINABOOH -1
#define ind_LHAROM -1
#define ind_CHF3 -1
#define ind_CHF2CF3 -1
#define ind_CH3CF3 -1
#define ind_CH2F2 -1
#define ind_CH3CHF2 -1
#define ind_CF2ClCF2Cl -1
#define ind_CF2ClCFCl2 -1
#define ind_CF3CF2Cl -1
#define ind_CH2Cl2 -1
#define ind_CH2FCF3 -1
#define ind_CH3CFCl2 -1
#define ind_CHCl3 -1
#define ind_CHF2Cl -1
#define ind_LBROMINE -1
#define ind_C3H7I -1
#define ind_CH2ClI -1
#define ind_CH2I2 -1
#define ind_HI -1
#define ind_HIO3 -1
#define ind_HOI -1
#define ind_I -1
#define ind_I2 -1
#define ind_I2O2 -1
#define ind_IBr -1
#define ind_ICl -1
#define ind_INO2 -1
#define ind_INO3 -1
#define ind_IO -1
#define ind_IPART -1
#define ind_OIO -1
#define ind_OCS -1
#define ind_S -1
#define ind_SF6 -1
#define ind_SH -1
#define ind_SO -1
#define ind_SO3 -1
#define ind_LSULFUR -1
#define ind_Hg -1
#define ind_HgO -1
#define ind_HgCl -1
#define ind_HgCl2 -1
#define ind_HgBr -1
#define ind_HgBr2 -1
#define ind_ClHgBr -1
#define ind_BrHgOBr -1
#define ind_ClHgOBr -1
#define ind_RGM_cs -1
#define ind_PRODUCTS -1
#define ind_M -1
#define ind_Op -1
#define ind_O2p -1
#define ind_Np -1
#define ind_N2p -1
#define ind_NOp -1
#define ind_em -1
#define ind_kJmol -1
#define ind_O4Sp -1
#define ind_O2Dp -1
#define ind_O2Pp -1
#define ind_LTERP -1
#define ind_LALK4 -1
#define ind_LALK5 -1
#define ind_LARO1 -1
#define ind_LARO2 -1
#define ind_LOLE1 -1
#define ind_LOLE2 -1
#define ind_LfPOG02 -1
#define ind_LfPOG03 -1
#define ind_LfPOG04 -1
#define ind_LfPOG05 -1
#define ind_LbbPOG02 -1
#define ind_LbbPOG03 -1
#define ind_LbbPOG04 -1
#define ind_LfSOGsv01 -1
#define ind_LfSOGsv02 -1
#define ind_LbbSOGsv01 -1
#define ind_LbbSOGsv02 -1
#define ind_LfSOGiv01 -1
#define ind_LfSOGiv02 -1
#define ind_LfSOGiv03 -1
#define ind_LfSOGiv04 -1
#define ind_LbbSOGiv01 -1
#define ind_LbbSOGiv02 -1
#define ind_LbbSOGiv03 -1
#define ind_LbSOGv01 -1
#define ind_LbSOGv02 -1
#define ind_LbSOGv03 -1
#define ind_LbSOGv04 -1
#define ind_LbOSOGv01 -1
#define ind_LbOSOGv02 -1
#define ind_LbOSOGv03 -1
#define ind_LaSOGv01 -1
#define ind_LaSOGv02 -1
#define ind_LaSOGv03 -1
#define ind_LaSOGv04 -1
#define ind_LaOSOGv01 -1
#define ind_LaOSOGv02 -1
#define ind_LaOSOGv03 -1
#define ind_ACBZO2 -1
#define ind_ALKNO3 -1
#define ind_ALKO2 -1
#define ind_ALKOH -1
#define ind_ALKOOH -1
#define ind_BCARY -1
#define ind_BENZO2 -1
#define ind_BENZOOH -1
#define ind_BEPOMUC -1
#define ind_BIGALD1 -1
#define ind_BIGALD2 -1
#define ind_BIGALD3 -1
#define ind_BIGALD4 -1
#define ind_BIGALKANE -1
#define ind_BIGENE -1
#define ind_BrONO -1
#define ind_BZALD -1
#define ind_BZOO -1
#define ind_BZOOH -1
#define ind_C3H7O2 -1
#define ind_C3H7OOH -1
#define ind_CFC113 -1
#define ind_CFC114 -1
#define ind_CFC115 -1
#define ind_COF2 -1
#define ind_COFCL -1
#define ind_DICARBO2 -1
#define ind_ELVOC -1
#define ind_ENEO2 -1
#define ind_EOOH -1
#define ind_F -1
#define ind_H1202 -1
#define ind_H2402 -1
#define ind_HCFC141B -1
#define ind_HCFC142B -1
#define ind_HCFC22 -1
#define ind_HF -1
#define ind_HOCH2OO -1
#define ind_HPALD -1
#define ind_IEC1O2 -1
#define ind_LIECHO -1
#define ind_LIECO3 -1
#define ind_LIECO3H -1
#define ind_LIMON -1
#define ind_LISOPNO3NO3 -1
#define ind_LISOPNO3O2 -1
#define ind_LISOPNO3OOH -1
#define ind_LISOPOOHO2 -1
#define ind_LISOPOOHOOH -1
#define ind_MALO2 -1
#define ind_MBONO3O2 -1
#define ind_MBOO2 -1
#define ind_MBOOOH -1
#define ind_MDIALO2 -1
#define ind_MEKNO3 -1
#define ind_MVKN -1
#define ind_MYRC -1
#define ind_NTERPNO3 -1
#define ind_NTERPO2 -1
#define ind_PACALD -1
#define ind_PBZNIT -1
#define ind_TEPOMUC -1
#define ind_TERP2O2 -1
#define ind_TERP2OOH -1
#define ind_TERPNO3 -1
#define ind_TERPO2 -1
#define ind_TERPOOH -1
#define ind_TERPROD1 -1
#define ind_TERPROD2 -1
#define ind_TOLO2 -1
#define ind_TOLOOH -1
#define ind_XYLENO2 -1
#define ind_XYLENOOH -1
#define ind_XYLOL -1
#define ind_XYLOLO2 -1
#define ind_XYLOLOOH -1
#define ind_O2_1D -1
#define ind_O2_1S -1
#define ind_ONIT -1
#define ind_C4H8 -1
#define ind_C4H9O3 -1
#define ind_C5H12 -1
#define ind_C5H11O2 -1
#define ind_C5H6O2 -1
#define ind_HYDRALD -1
#define ind_ISOPO2 -1
#define ind_C5H9O3 -1
#define ind_ISOPOOH -1
#define ind_C5H12O2 -1
#define ind_ONITR -1
#define ind_C5H10O4 -1
#define ind_ROO6R5P -1
#define ind_NH4 -1
#define ind_SO4 -1
#define ind_HCO -1
#define ind_ISPD -1
#define ind_ClOO -1
#define ind_Rn -1
#define ind_Pb -1
#define ind_XO2 -1
#define ind_XO2N -1
#define ind_ROOH -1
#define ind_OLE -1
#define ind_ROR -1
#define ind_ORGNTR -1
#define ind_ACO2 -1
#define ind_PAR -1
#define ind_RXPAR -1
#define ind_OHv0 -1
#define ind_OHv1 -1
#define ind_OHv2 -1
#define ind_OHv3 -1
#define ind_OHv4 -1
#define ind_OHv5 -1
#define ind_OHv6 -1
#define ind_OHv7 -1
#define ind_OHv8 -1
#define ind_OHv9 -1
#define ind_O1S -1
#define ind_O21d -1
#define ind_O2b1s -1
#define ind_O2c1s -1
#define ind_O2x -1
#define ind_O2A3D -1
#define ind_O2A3S -1
#define ind_O25P -1
#define ind_O2_a01 -1
#define ind_O3_a01 -1
#define ind_OH_a01 -1
#define ind_HO2_a01 -1
#define ind_H2O_a01 -1
#define ind_H2O2_a01 -1
#define ind_NH3_a01 -1
#define ind_NO_a01 -1
#define ind_NO2_a01 -1
#define ind_NO3_a01 -1
#define ind_HONO_a01 -1
#define ind_HNO3_a01 -1
#define ind_HNO4_a01 -1
#define ind_CH3OH_a01 -1
#define ind_HCOOH_a01 -1
#define ind_HCHO_a01 -1
#define ind_CH3O2_a01 -1
#define ind_CH3OOH_a01 -1
#define ind_CO2_a01 -1
#define ind_CH3CO2H_a01 -1
#define ind_PAN_a01 -1
#define ind_CH3CHO_a01 -1
#define ind_CH3COCH3_a01 -1
#define ind_Cl_a01 -1
#define ind_Cl2_a01 -1
#define ind_HCl_a01 -1
#define ind_HOCl_a01 -1
#define ind_Br_a01 -1
#define ind_Br2_a01 -1
#define ind_HBr_a01 -1
#define ind_HOBr_a01 -1
#define ind_BrCl_a01 -1
#define ind_I2_a01 -1
#define ind_IO_a01 -1
#define ind_HOI_a01 -1
#define ind_ICl_a01 -1
#define ind_IBr_a01 -1
#define ind_SO2_a01 -1
#define ind_H2SO4_a01 -1
#define ind_DMS_a01 -1
#define ind_DMSO_a01 -1
#define ind_Hg_a01 -1
#define ind_HgO_a01 -1
#define ind_HgOHOH_a01 -1
#define ind_HgOHCl_a01 -1
#define ind_HgCl2_a01 -1
#define ind_HgBr2_a01 -1
#define ind_HgSO3_a01 -1
#define ind_ClHgBr_a01 -1
#define ind_BrHgOBr_a01 -1
#define ind_ClHgOBr_a01 -1
#define ind_FeOH3_a01 -1
#define ind_FeCl3_a01 -1
#define ind_FeF3_a01 -1
#define ind_O2m_a01 -1
#define ind_OHm_a01 -1
#define ind_HO2m_a01 -1
#define ind_O2mm_a01 -1
#define ind_Hp_a01 -1
#define ind_NH4p_a01 -1
#define ind_NO2m_a01 -1
#define ind_NO3m_a01 -1
#define ind_NO4m_a01 -1
#define ind_CO3m_a01 -1
#define ind_HCOOm_a01 -1
#define ind_HCO3m_a01 -1
#define ind_CH3COOm_a01 -1
#define ind_Clm_a01 -1
#define ind_Cl2m_a01 -1
#define ind_ClOm_a01 -1
#define ind_ClOHm_a01 -1
#define ind_Brm_a01 -1
#define ind_Br2m_a01 -1
#define ind_BrOm_a01 -1
#define ind_BrOHm_a01 -1
#define ind_BrCl2m_a01 -1
#define ind_Br2Clm_a01 -1
#define ind_Im_a01 -1
#define ind_IO2m_a01 -1
#define ind_IO3m_a01 -1
#define ind_ICl2m_a01 -1
#define ind_IBr2m_a01 -1
#define ind_SO3m_a01 -1
#define ind_SO3mm_a01 -1
#define ind_SO4m_a01 -1
#define ind_SO4mm_a01 -1
#define ind_SO5m_a01 -1
#define ind_HSO3m_a01 -1
#define ind_HSO4m_a01 -1
#define ind_HSO5m_a01 -1
#define ind_CH3SO3m_a01 -1
#define ind_CH2OHSO3m_a01 -1
#define ind_Hgp_a01 -1
#define ind_Hgpp_a01 -1
#define ind_HgOHp_a01 -1
#define ind_HgClp_a01 -1
#define ind_HgBrp_a01 -1
#define ind_HgSO32mm_a01 -1
#define ind_Fepp_a01 -1
#define ind_FeOpp_a01 -1
#define ind_FeOHp_a01 -1
#define ind_FeOH2p_a01 -1
#define ind_FeClp_a01 -1
#define ind_Feppp_a01 -1
#define ind_FeHOpp_a01 -1
#define ind_FeHO2pp_a01 -1
#define ind_FeOHpp_a01 -1
#define ind_FeOH4m_a01 -1
#define ind_FeOHHO2p_a01 -1
#define ind_FeClpp_a01 -1
#define ind_FeCl2p_a01 -1
#define ind_FeBrpp_a01 -1
#define ind_FeBr2p_a01 -1
#define ind_FeFpp_a01 -1
#define ind_FeF2p_a01 -1
#define ind_FeSO3p_a01 -1
#define ind_FeSO4p_a01 -1
#define ind_FeSO42m_a01 -1
#define ind_FeOH2Fepppp_a01 -1
#define ind_D1O_a01 -1
#define ind_Nap_a01 -1
#define ind_LossO3Su -1

#define ihs_N2O5_H2O 0
#define ihs_HOCl_HCl 1
#define ihs_ClNO3_HCl 2
#define ihs_ClNO3_H2O 3
#define ihs_N2O5_HCl 4
#define ihs_ClNO3_HBr 5
#define ihs_BrNO3_HCl 6
#define ihs_HOCl_HBr 7
#define ihs_HOBr_HCl 8
#define ihs_HOBr_HBr 9
#define ihs_BrNO3_H2O 10
#define ihs_Hg 11
#define ihs_RGM 12

#define iht_N2O5 0
#define iht_HNO3 1
#define iht_Hg 2
#define iht_RGM 3

#define k_C6H5O_NO2    (2.08E-12)
#define k_C6H5O_O3     (2.86E-13)
#define k_adsecprim   (3.0E-11)
#define k_adtertprim   (5.7E-11 )
#define f_soh   (3.44)
#define f_toh   (2.68)
#define f_sooh   (7.)
#define f_tooh   (7.)
#define f_ono2   (0.04 )
#define f_ch2ono2   (0.2)
#define f_cpan  (.25)
#define f_allyl   (3.6)
#define f_alk  (1.23)
#define f_cho   (0.55)
#define f_co2h   (1.67)
#define f_co   (0.73)
#define f_o   (8.15)
#define f_pch2oh   (1.29)
#define f_tch2oh   (0.53)
#define a_pan   (0.56      )
#define a_cho   (0.31   )
#define a_coch3   (0.76 )
#define a_ch2ono2   (0.64  )
#define a_ch2oh   (1.7  )
#define a_ch2ooh   (1.7 )
#define a_coh   (2.2       )
#define a_cooh   (2.2   )
#define a_co2h   (0.25)

#define ifun 0
#define ijac 1
#define istp 2
#define iacc 3
#define irej 4
#define idec 5
#define isol 6
#define isng 7
#define itexit 0
#define ihexit 1

#define ZERO 0.0
#define ONE 1.0
#define HALF 0.5


/*
 * Fortran to C macros 
 * GPU-friendly array deffinition 
 * i:VL_GLO, j:NVAR 
 *
 */
#define conc(i,j)    conc[(j)*VL_GLO+(i)]
#define khet_st(i,j) khet_st[(j)*VL_GLO+(i)]
#define khet_tr(i,j) khet_tr[(j)*VL_GLO+(i)]
#define jx(i,j)      jx[j*VL_GLO+i]
#define istatus(i,j) istatus[(j)*(VL_GLO)+(i)]   
#define rstatus(i,j) rstatus[(j)*(VL_GLO)+(i)]


#define ROUND128(X)  (X + (128 - 1)) & ~(128 - 1)

#define rconst(i,j)  rconst[(j)]


/* Temporary arrays allocated in stack */
#define var(i,j)     var[(j)]
#define fix(i,j)     fix[(j)]
#define jcb(i,j)     jcb[(j)]
#define varDot(i,j)  varDot[j]
#define varNew(i,j) varNew[(j)]
#define Fcn0(i,j)   Fcn0[(j)]
#define Fcn(i,j)    Fcn[(j)]
#define Fcn(i,j)    Fcn[(j)]
#define dFdT(i,j)   dFdT[(j)]
#define varErr(i,j) varErr[(j)]
#define K(i,j,k) K[(j)*(NVAR)+(k)]
#define jac0(i,j)    jac0[(j)]    
#define Ghimj(i,j)   Ghimj[(j)]   

//#define var(i,j)     var[(j)]
//#define fix(i,j)     fix[(j)]
//#define jcb(i,j)     jcb[(j)]
//#define varDot(i,j)  varDot[j]
//#define varNew(i,j) varNew[(j)]
//#define Fcn0(i,j)   Fcn0[(j)]
//#define Fcn(i,j)    Fcn[(j)]
//#define Fcn(i,j)    Fcn[(j)]
//#define dFdT(i,j)   dFdT[(j)]
//#define varErr(i,j) varErr[(j)]
//#define K(i,j,k) K[(j)*(NVAR)+(k)]
//#define jac0(i,j)    jac0[(j)]    
//#define Ghimj(i,j)   Ghimj[(j)]   


/* Enable debug flags for GPU */
//#define DEBUG

#ifdef DEBUG
#define GPU_DEBUG()\
    gpuErrchk( cudaPeekAtLastError()   ); \
    gpuErrchk( cudaDeviceSynchronize() ); 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#else 
/* If debug flags are disabled */
#define GPU_DEBUG()
#define gpuErrchk(ans) ans
#endif

/** prefetches into L1 cache */
__device__ inline void prefetch_gl1(const void *p) {
#if __CUDA_ARCH__ <= 300
        asm("prefetch.global.L1 [%0];": :"l"(p));
#endif
}
__device__ inline void prefetch_ll1(const void *p) {
#if __CUDA_ARCH__ <= 300
        asm("prefetch.local.L1 [%0];": :"l"(p));
#endif
}

/** prefetches into L2 cache */
__device__ inline void prefetch_gl2(const void *p) {
#if __CUDA_ARCH__ <= 300
        asm("prefetch.global.L2 [%0];": :"l"(p));
#endif
}
__device__ inline void prefetch_ll2(const void *p) {
#if __CUDA_ARCH__ <= 300
        asm("prefetch.local.L2 [%0];": :"l"(p));
#endif
}



__device__ void  update_rconst(const double * __restrict__ var,
			       const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
			       const double * __restrict__ jx, double * __restrict__ rconst,
 			       const double * __restrict__ temp_gpu,
 			       const double * __restrict__ press_gpu,
 			       const double * __restrict__ cair_gpu,
			       const int VL_GLO);

/* This runs on CPU */
double machine_eps_flt()
{
    double machEps = 1.0f;

    do
    {
        machEps /= 2.0f;
        // If next epsilon yields 1, then break, because current
        // epsilon is the machine epsilon.
    }
    while ((double)(1.0 + (machEps/2.0)) != 1.0);

    return machEps;
}

/* This runs on GPU */
__device__ double machine_eps_flt_cuda() 
{
    typedef union 
    {
        long  i64;
        double f64;
    } flt_64;

    flt_64 s;

    s.f64 = 1.;
    s.i64++;
    return (s.f64 - 1.);
}

__device__  static double alpha_AN(const int n, const int ro2type, const double temp, const double cair){
    double alpha=2.E-22, beta=1.0, Yinf_298K=0.43,  F=0.41, m0=0., minf=8.0;
    double Y0_298K, Y0_298K_tp, Yinf_298K_t, zeta, k_ratio, alpha_a;
    /*  IF (ro2type = 1) THEN   m = 0.4                !   primary RO2
        ELSE IF (ro2type = 2) THEN  m = 1.                 !   secondary RO2
        ELSE IF (ro2type = 3) THEN  m = 0.3                !   tertiary RO2
        ELSE  m = 1.
  */
    double m = 1.;
    Y0_298K     = alpha*exp(beta*n);
    Y0_298K_tp  = Y0_298K *cair *pow((temp/298.),(- m0));
    Yinf_298K_t = Yinf_298K * pow((temp/298.),(- minf));
    zeta        = 1/(1+ pow(log10(Y0_298K_tp/Yinf_298K_t),2));
    k_ratio     = (Y0_298K_tp/(1+ Y0_298K_tp/Yinf_298K_t))*pow(F,zeta);
    alpha_a    = k_ratio/(1+ k_ratio) *m;
    return alpha_a;
}
__device__  static double alpha_AN(const int n, const int ro2type, const int bcarb, const int gcarb, const int abic, const double temp, const double cair){
    double alpha=2.E-22, beta=1.0, Yinf_298K=0.43,  F=0.41, m0=0., minf=8.0;
    double Y0_298K, Y0_298K_tp, Yinf_298K_t, zeta, k_ratio, alpha_a;
    double bcf=1., gcf=1., abf=1.;
    double m = 1.; //According to Teng, ref3189

if (bcarb == 1) { bcf = 0.19; }// derived from Praske, ref3190: alpha_AN = 0.03 for the secondary HMKO2 relative to alpha_AN for 6C RO2 (0.16)
if (gcarb == 1) {gcf = 0.44; }// derived from Praske, ref3190: alpha_AN = 0.07 for the primary HMKO2 relative to alpha_AN for 6C RO2 (0.16)
if (abic == 1) { abf = 0.24; }// derived from the ratio of AN- yield for toluene from Elrod et al. (ref3180), 5.5 0x1.9206e69676542p+ 229t & 
                              // 200 torr, and this SAR for linear alkyl RO2 with 9 heavy atoms, 23.3%

    Y0_298K     = alpha*exp(beta*n);
    Y0_298K_tp  = Y0_298K *cair *pow((temp/298.),(- m0));
    Yinf_298K_t = Yinf_298K * pow((temp/298.),(- minf));
    zeta        = 1/(1+ pow(log10(Y0_298K_tp/Yinf_298K_t),2));
    k_ratio     = (Y0_298K_tp/(1+ Y0_298K_tp/Yinf_298K_t))*pow(F,zeta);
    alpha_a    = k_ratio/(1+ k_ratio) *m*bcf*gcf*abf;
    return alpha_a;
}
__device__  static double k_RO2_HO2(const double temp, const int nC){
    return 2.91e-13*exp(1300./temp)*(1.-exp(-0.245*nC)); // ref1630
}
__device__ double ros_ErrorNorm(double * __restrict__ var, double * __restrict__ varNew, double * __restrict__ varErr, 
                                const double * __restrict__ absTol, const double * __restrict__ relTol,
                                const int vectorTol )
{
    double err, scale, varMax;

    int index = blockIdx.x*blockDim.x+threadIdx.x;

    err = ZERO;

    if (vectorTol){
        for (int i=0;i<NVAR - 16;i+=16){
            prefetch_ll1(&varErr(index,i));
            prefetch_ll1(&absTol[i]);
            prefetch_ll1(&relTol[i]);
            prefetch_ll1(&var(index,i));
            prefetch_ll1(&varNew(index,i));
        }

        for (int i=0; i<NVAR; i++)
        {
            varMax = fmax(fabs(var(index,i)),fabs(varNew(index,i)));
            scale = absTol[i]+ relTol[i]*varMax;

            err += pow((double)varErr(index,i)/scale,2.0);
        }
        err  = sqrt((double) err/NVAR);
    }else{
        for (int i=0;i<NVAR - 16;i+=16){
            prefetch_ll1(&varErr(index,i));
            prefetch_ll1(&var(index,i));
            prefetch_ll1(&varNew(index,i));
        }

        for (int i=0; i<NVAR; i++)
        {
            varMax = fmax(fabs(var(index,i)),fabs(varNew(index,i)));

            scale = absTol[0]+ relTol[0]*varMax;
            err += pow((double)varErr(index,i)/scale,2.0);
        }
        err  = sqrt((double) err/NVAR);
    }

    return err;


}

__device__ void kppSolve(const double * __restrict__ Ghimj, double * __restrict__ K, 
                         const int istage, const int ros_S ){
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    //K = &K[istage*NVAR];

        K(index,istage,7) = K(index,istage,7)- Ghimj(index,7)*K(index,istage,1)- Ghimj(index,8)*K(index,istage,2);
        K(index,istage,8) = K(index,istage,8)- Ghimj(index,23)*K(index,istage,1)- Ghimj(index,24)*K(index,istage,2);
        K(index,istage,14) = K(index,istage,14)- Ghimj(index,50)*K(index,istage,5)- Ghimj(index,51)*K(index,istage,6);
        K(index,istage,19) = K(index,istage,19)- Ghimj(index,67)*K(index,istage,4);
        K(index,istage,31) = K(index,istage,31)- Ghimj(index,188)*K(index,istage,1)- Ghimj(index,189)*K(index,istage,2);
        K(index,istage,32) = K(index,istage,32)- Ghimj(index,193)*K(index,istage,1);
        K(index,istage,34) = K(index,istage,34)- Ghimj(index,205)*K(index,istage,0);
        K(index,istage,60) = K(index,istage,60)- Ghimj(index,309)*K(index,istage,59);
        K(index,istage,70) = K(index,istage,70)- Ghimj(index,351)*K(index,istage,61);
        K(index,istage,85) = K(index,istage,85)- Ghimj(index,426)*K(index,istage,79);
        K(index,istage,86) = K(index,istage,86)- Ghimj(index,434)*K(index,istage,62)- Ghimj(index,435)*K(index,istage,69);
        K(index,istage,87) = K(index,istage,87)- Ghimj(index,442)*K(index,istage,70)- Ghimj(index,443)*K(index,istage,84);
        K(index,istage,90) = K(index,istage,90)- Ghimj(index,468)*K(index,istage,80);
        K(index,istage,92) = K(index,istage,92)- Ghimj(index,487)*K(index,istage,47)- Ghimj(index,488)*K(index,istage,84);
        K(index,istage,93) = K(index,istage,93)- Ghimj(index,495)*K(index,istage,49)- Ghimj(index,496)*K(index,istage,69);
        K(index,istage,94) = K(index,istage,94)- Ghimj(index,502)*K(index,istage,72)- Ghimj(index,503)*K(index,istage,86)- Ghimj(index,504)*K(index,istage,93);
        K(index,istage,95) = K(index,istage,95)- Ghimj(index,510)*K(index,istage,58)- Ghimj(index,511)*K(index,istage,77)- Ghimj(index,512)*K(index,istage,82)- Ghimj(index,513)*K(index,istage,91);
        K(index,istage,96) = K(index,istage,96)- Ghimj(index,535)*K(index,istage,72)- Ghimj(index,536)*K(index,istage,82)- Ghimj(index,537)*K(index,istage,94);
        K(index,istage,99) = K(index,istage,99)- Ghimj(index,563)*K(index,istage,68)- Ghimj(index,564)*K(index,istage,85);
        K(index,istage,100) = K(index,istage,100)- Ghimj(index,572)*K(index,istage,90);
        K(index,istage,101) = K(index,istage,101)- Ghimj(index,585)*K(index,istage,83);
        K(index,istage,102) = K(index,istage,102)- Ghimj(index,598)*K(index,istage,40)- Ghimj(index,599)*K(index,istage,79);
        K(index,istage,108) = K(index,istage,108)- Ghimj(index,630)*K(index,istage,64)- Ghimj(index,631)*K(index,istage,67)- Ghimj(index,632)*K(index,istage,82)- Ghimj(index,633)*K(index,istage,91)- Ghimj(index,634)*K(index,istage,94)- Ghimj(index,635)*K(index,istage,106);
        K(index,istage,109) = K(index,istage,109)- Ghimj(index,647)*K(index,istage,106);
        K(index,istage,110) = K(index,istage,110)- Ghimj(index,655)*K(index,istage,66)- Ghimj(index,656)*K(index,istage,91)- Ghimj(index,657)*K(index,istage,106)- Ghimj(index,658)*K(index,istage,109);
        K(index,istage,111) = K(index,istage,111)- Ghimj(index,666)*K(index,istage,99)- Ghimj(index,667)*K(index,istage,102)- Ghimj(index,668)*K(index,istage,107);
        K(index,istage,113) = K(index,istage,113)- Ghimj(index,685)*K(index,istage,64)- Ghimj(index,686)*K(index,istage,82)- Ghimj(index,687)*K(index,istage,106)- Ghimj(index,688)*K(index,istage,110);
        K(index,istage,115) = K(index,istage,115)- Ghimj(index,703)*K(index,istage,67)- Ghimj(index,704)*K(index,istage,103)- Ghimj(index,705)*K(index,istage,107);
        K(index,istage,117) = K(index,istage,117)- Ghimj(index,722)*K(index,istage,48)- Ghimj(index,723)*K(index,istage,49)- Ghimj(index,724)*K(index,istage,71)- Ghimj(index,725)*K(index,istage,79)- Ghimj(index,726)*K(index,istage,85)- Ghimj(index,727)*K(index,istage,102)- Ghimj(index,728)  *K(index,istage,107)- Ghimj(index,729)*K(index,istage,111)- Ghimj(index,730)*K(index,istage,115);
        K(index,istage,118) = K(index,istage,118)- Ghimj(index,741)*K(index,istage,100)- Ghimj(index,742)*K(index,istage,105)- Ghimj(index,743)*K(index,istage,112)- Ghimj(index,744)*K(index,istage,116);
        K(index,istage,119) = K(index,istage,119)- Ghimj(index,758)*K(index,istage,68)- Ghimj(index,759)*K(index,istage,71)- Ghimj(index,760)*K(index,istage,79)- Ghimj(index,761)*K(index,istage,99)- Ghimj(index,762)*K(index,istage,102)- Ghimj(index,763)*K(index,istage,107)- Ghimj(index,764)  *K(index,istage,111)- Ghimj(index,765)*K(index,istage,115)- Ghimj(index,766)*K(index,istage,117);
        K(index,istage,120) = K(index,istage,120)- Ghimj(index,777)*K(index,istage,41)- Ghimj(index,778)*K(index,istage,42)- Ghimj(index,779)*K(index,istage,43)- Ghimj(index,780)*K(index,istage,57)- Ghimj(index,781)*K(index,istage,60)- Ghimj(index,782)*K(index,istage,75)- Ghimj(index,783)  *K(index,istage,92)- Ghimj(index,784)*K(index,istage,97)- Ghimj(index,785)*K(index,istage,98)- Ghimj(index,786)*K(index,istage,107);
        K(index,istage,121) = K(index,istage,121)- Ghimj(index,798)*K(index,istage,38)- Ghimj(index,799)*K(index,istage,63)- Ghimj(index,800)*K(index,istage,68)- Ghimj(index,801)*K(index,istage,72)- Ghimj(index,802)*K(index,istage,77)- Ghimj(index,803)*K(index,istage,82)- Ghimj(index,804)  *K(index,istage,85)- Ghimj(index,805)*K(index,istage,86)- Ghimj(index,806)*K(index,istage,93)- Ghimj(index,807)*K(index,istage,94)- Ghimj(index,808)*K(index,istage,96)- Ghimj(index,809)*K(index,istage,99)- Ghimj(index,810)*K(index,istage,102)- Ghimj(index,811) *K(index,istage,106)- Ghimj(index,812)*K(index,istage,107)- Ghimj(index,813)*K(index,istage,108)- Ghimj(index,814)*K(index,istage,109)- Ghimj(index,815)*K(index,istage,110)- Ghimj(index,816)*K(index,istage,111)- Ghimj(index,817)*K(index,istage,113) - Ghimj(index,818)*K(index,istage,115)- Ghimj(index,819)*K(index,istage,117)- Ghimj(index,820)*K(index,istage,119);
        K(index,istage,122) = K(index,istage,122)- Ghimj(index,831)*K(index,istage,75)- Ghimj(index,832)*K(index,istage,95)- Ghimj(index,833)*K(index,istage,96)- Ghimj(index,834)*K(index,istage,97)- Ghimj(index,835)*K(index,istage,98)- Ghimj(index,836)*K(index,istage,103)- Ghimj(index,837)  *K(index,istage,106)- Ghimj(index,838)*K(index,istage,107)- Ghimj(index,839)*K(index,istage,108)- Ghimj(index,840)*K(index,istage,109)- Ghimj(index,841)*K(index,istage,110)- Ghimj(index,842)*K(index,istage,113)- Ghimj(index,843)*K(index,istage,115) - Ghimj(index,844)*K(index,istage,119)- Ghimj(index,845)*K(index,istage,120)- Ghimj(index,846)*K(index,istage,121);
        K(index,istage,123) = K(index,istage,123)- Ghimj(index,861)*K(index,istage,103)- Ghimj(index,862)*K(index,istage,104)- Ghimj(index,863)*K(index,istage,112)- Ghimj(index,864)*K(index,istage,114)- Ghimj(index,865)*K(index,istage,116)- Ghimj(index,866)*K(index,istage,118)  - Ghimj(index,867)*K(index,istage,119)- Ghimj(index,868)*K(index,istage,121);
        K(index,istage,124) = K(index,istage,124)- Ghimj(index,885)*K(index,istage,81)- Ghimj(index,886)*K(index,istage,84)- Ghimj(index,887)*K(index,istage,92)- Ghimj(index,888)*K(index,istage,103)- Ghimj(index,889)*K(index,istage,106)- Ghimj(index,890)*K(index,istage,107)- Ghimj(index,891)  *K(index,istage,110)- Ghimj(index,892)*K(index,istage,114)- Ghimj(index,893)*K(index,istage,120)- Ghimj(index,894)*K(index,istage,121)- Ghimj(index,895)*K(index,istage,122);
        K(index,istage,125) = K(index,istage,125)- Ghimj(index,910)*K(index,istage,3)- Ghimj(index,911)*K(index,istage,53)- Ghimj(index,912)*K(index,istage,63)- Ghimj(index,913)*K(index,istage,65)- Ghimj(index,914)*K(index,istage,74)- Ghimj(index,915)*K(index,istage,75)- Ghimj(index,916)  *K(index,istage,81)- Ghimj(index,917)*K(index,istage,86)- Ghimj(index,918)*K(index,istage,93)- Ghimj(index,919)*K(index,istage,94)- Ghimj(index,920)*K(index,istage,98)- Ghimj(index,921)*K(index,istage,102)- Ghimj(index,922)*K(index,istage,104)- Ghimj(index,923) *K(index,istage,106)- Ghimj(index,924)*K(index,istage,107)- Ghimj(index,925)*K(index,istage,109)- Ghimj(index,926)*K(index,istage,113)- Ghimj(index,927)*K(index,istage,114)- Ghimj(index,928)*K(index,istage,117)- Ghimj(index,929)*K(index,istage,119) - Ghimj(index,930)*K(index,istage,120)- Ghimj(index,931)*K(index,istage,121)- Ghimj(index,932)*K(index,istage,122)- Ghimj(index,933)*K(index,istage,124);
        K(index,istage,126) = K(index,istage,126)- Ghimj(index,948)*K(index,istage,40)- Ghimj(index,949)*K(index,istage,44)- Ghimj(index,950)*K(index,istage,45)- Ghimj(index,951)*K(index,istage,47)- Ghimj(index,952)*K(index,istage,48)- Ghimj(index,953)*K(index,istage,49)- Ghimj(index,954)  *K(index,istage,52)- Ghimj(index,955)*K(index,istage,53)- Ghimj(index,956)*K(index,istage,54)- Ghimj(index,957)*K(index,istage,55)- Ghimj(index,958)*K(index,istage,56)- Ghimj(index,959)*K(index,istage,57)- Ghimj(index,960)*K(index,istage,58)- Ghimj(index,961) *K(index,istage,61)- Ghimj(index,962)*K(index,istage,62)- Ghimj(index,963)*K(index,istage,63)- Ghimj(index,964)*K(index,istage,64)- Ghimj(index,965)*K(index,istage,65)- Ghimj(index,966)*K(index,istage,66)- Ghimj(index,967)*K(index,istage,67)- Ghimj(index,968) *K(index,istage,68)- Ghimj(index,969)*K(index,istage,69)- Ghimj(index,970)*K(index,istage,70)- Ghimj(index,971)*K(index,istage,71)- Ghimj(index,972)*K(index,istage,72)- Ghimj(index,973)*K(index,istage,73)- Ghimj(index,974)*K(index,istage,74)- Ghimj(index,975) *K(index,istage,75)- Ghimj(index,976)*K(index,istage,76)- Ghimj(index,977)*K(index,istage,77)- Ghimj(index,978)*K(index,istage,78)- Ghimj(index,979)*K(index,istage,79)- Ghimj(index,980)*K(index,istage,81)- Ghimj(index,981)*K(index,istage,82)- Ghimj(index,982) *K(index,istage,84)- Ghimj(index,983)*K(index,istage,85)- Ghimj(index,984)*K(index,istage,86)- Ghimj(index,985)*K(index,istage,87)- Ghimj(index,986)*K(index,istage,88)- Ghimj(index,987)*K(index,istage,89)- Ghimj(index,988)*K(index,istage,91)- Ghimj(index,989) *K(index,istage,92)- Ghimj(index,990)*K(index,istage,93)- Ghimj(index,991)*K(index,istage,94)- Ghimj(index,992)*K(index,istage,95)- Ghimj(index,993)*K(index,istage,96)- Ghimj(index,994)*K(index,istage,97)- Ghimj(index,995)*K(index,istage,98)- Ghimj(index,996) *K(index,istage,99)- Ghimj(index,997)*K(index,istage,100)- Ghimj(index,998)*K(index,istage,101)- Ghimj(index,999)*K(index,istage,102)- Ghimj(index,1000)*K(index,istage,103)- Ghimj(index,1001)*K(index,istage,104)- Ghimj(index,1002)*K(index,istage,105) - Ghimj(index,1003)*K(index,istage,106)- Ghimj(index,1004)*K(index,istage,107)- Ghimj(index,1005)*K(index,istage,108)- Ghimj(index,1006)*K(index,istage,109)- Ghimj(index,1007)*K(index,istage,110)- Ghimj(index,1008)*K(index,istage,111) - Ghimj(index,1009)*K(index,istage,112)- Ghimj(index,1010)*K(index,istage,113)- Ghimj(index,1011)*K(index,istage,114)- Ghimj(index,1012)*K(index,istage,115)- Ghimj(index,1013)*K(index,istage,116)- Ghimj(index,1014)*K(index,istage,117) - Ghimj(index,1015)*K(index,istage,118)- Ghimj(index,1016)*K(index,istage,119)- Ghimj(index,1017)*K(index,istage,120)- Ghimj(index,1018)*K(index,istage,121)- Ghimj(index,1019)*K(index,istage,122)- Ghimj(index,1020)*K(index,istage,123) - Ghimj(index,1021)*K(index,istage,124)- Ghimj(index,1022)*K(index,istage,125);
        K(index,istage,127) = K(index,istage,127)- Ghimj(index,1036)*K(index,istage,1)- Ghimj(index,1037)*K(index,istage,39)- Ghimj(index,1038)*K(index,istage,41)- Ghimj(index,1039)*K(index,istage,42)- Ghimj(index,1040)*K(index,istage,43)- Ghimj(index,1041)*K(index,istage,50)  - Ghimj(index,1042)*K(index,istage,52)- Ghimj(index,1043)*K(index,istage,54)- Ghimj(index,1044)*K(index,istage,55)- Ghimj(index,1045)*K(index,istage,57)- Ghimj(index,1046)*K(index,istage,75)- Ghimj(index,1047)*K(index,istage,80)- Ghimj(index,1048) *K(index,istage,83)- Ghimj(index,1049)*K(index,istage,88)- Ghimj(index,1050)*K(index,istage,90)- Ghimj(index,1051)*K(index,istage,97)- Ghimj(index,1052)*K(index,istage,98)- Ghimj(index,1053)*K(index,istage,100)- Ghimj(index,1054)*K(index,istage,103) - Ghimj(index,1055)*K(index,istage,104)- Ghimj(index,1056)*K(index,istage,105)- Ghimj(index,1057)*K(index,istage,106)- Ghimj(index,1058)*K(index,istage,107)- Ghimj(index,1059)*K(index,istage,112)- Ghimj(index,1060)*K(index,istage,114) - Ghimj(index,1061)*K(index,istage,116)- Ghimj(index,1062)*K(index,istage,118)- Ghimj(index,1063)*K(index,istage,119)- Ghimj(index,1064)*K(index,istage,120)- Ghimj(index,1065)*K(index,istage,121)- Ghimj(index,1066)*K(index,istage,122) - Ghimj(index,1067)*K(index,istage,123)- Ghimj(index,1068)*K(index,istage,124)- Ghimj(index,1069)*K(index,istage,125)- Ghimj(index,1070)*K(index,istage,126);
        K(index,istage,128) = K(index,istage,128)- Ghimj(index,1083)*K(index,istage,40)- Ghimj(index,1084)*K(index,istage,44)- Ghimj(index,1085)*K(index,istage,45)- Ghimj(index,1086)*K(index,istage,47)- Ghimj(index,1087)*K(index,istage,48)- Ghimj(index,1088)*K(index,istage,49)  - Ghimj(index,1089)*K(index,istage,52)- Ghimj(index,1090)*K(index,istage,53)- Ghimj(index,1091)*K(index,istage,54)- Ghimj(index,1092)*K(index,istage,55)- Ghimj(index,1093)*K(index,istage,57)- Ghimj(index,1094)*K(index,istage,61)- Ghimj(index,1095) *K(index,istage,63)- Ghimj(index,1096)*K(index,istage,67)- Ghimj(index,1097)*K(index,istage,70)- Ghimj(index,1098)*K(index,istage,73)- Ghimj(index,1099)*K(index,istage,74)- Ghimj(index,1100)*K(index,istage,75)- Ghimj(index,1101)*K(index,istage,76) - Ghimj(index,1102)*K(index,istage,77)- Ghimj(index,1103)*K(index,istage,78)- Ghimj(index,1104)*K(index,istage,79)- Ghimj(index,1105)*K(index,istage,83)- Ghimj(index,1106)*K(index,istage,84)- Ghimj(index,1107)*K(index,istage,86)- Ghimj(index,1108) *K(index,istage,87)- Ghimj(index,1109)*K(index,istage,88)- Ghimj(index,1110)*K(index,istage,92)- Ghimj(index,1111)*K(index,istage,93)- Ghimj(index,1112)*K(index,istage,97)- Ghimj(index,1113)*K(index,istage,98)- Ghimj(index,1114)*K(index,istage,101) - Ghimj(index,1115)*K(index,istage,102)- Ghimj(index,1116)*K(index,istage,103)- Ghimj(index,1117)*K(index,istage,104)- Ghimj(index,1118)*K(index,istage,105)- Ghimj(index,1119)*K(index,istage,106)- Ghimj(index,1120)*K(index,istage,107) - Ghimj(index,1121)*K(index,istage,110)- Ghimj(index,1122)*K(index,istage,111)- Ghimj(index,1123)*K(index,istage,112)- Ghimj(index,1124)*K(index,istage,114)- Ghimj(index,1125)*K(index,istage,115)- Ghimj(index,1126)*K(index,istage,116) - Ghimj(index,1127)*K(index,istage,117)- Ghimj(index,1128)*K(index,istage,118)- Ghimj(index,1129)*K(index,istage,119)- Ghimj(index,1130)*K(index,istage,120)- Ghimj(index,1131)*K(index,istage,121)- Ghimj(index,1132)*K(index,istage,122) - Ghimj(index,1133)*K(index,istage,123)- Ghimj(index,1134)*K(index,istage,124)- Ghimj(index,1135)*K(index,istage,125)- Ghimj(index,1136)*K(index,istage,126)- Ghimj(index,1137)*K(index,istage,127);
        K(index,istage,129) = K(index,istage,129)- Ghimj(index,1149)*K(index,istage,0)- Ghimj(index,1150)*K(index,istage,1)- Ghimj(index,1151)*K(index,istage,2)- Ghimj(index,1152)*K(index,istage,44)- Ghimj(index,1153)*K(index,istage,45)- Ghimj(index,1154)*K(index,istage,52)- Ghimj(index,1155)  *K(index,istage,53)- Ghimj(index,1156)*K(index,istage,54)- Ghimj(index,1157)*K(index,istage,55)- Ghimj(index,1158)*K(index,istage,80)- Ghimj(index,1159)*K(index,istage,90)- Ghimj(index,1160)*K(index,istage,100)- Ghimj(index,1161)*K(index,istage,103) - Ghimj(index,1162)*K(index,istage,104)- Ghimj(index,1163)*K(index,istage,105)- Ghimj(index,1164)*K(index,istage,112)- Ghimj(index,1165)*K(index,istage,114)- Ghimj(index,1166)*K(index,istage,116)- Ghimj(index,1167)*K(index,istage,118) - Ghimj(index,1168)*K(index,istage,119)- Ghimj(index,1169)*K(index,istage,121)- Ghimj(index,1170)*K(index,istage,123)- Ghimj(index,1171)*K(index,istage,124)- Ghimj(index,1172)*K(index,istage,125)- Ghimj(index,1173)*K(index,istage,126) - Ghimj(index,1174)*K(index,istage,127)- Ghimj(index,1175)*K(index,istage,128);
        K(index,istage,130) = K(index,istage,130)- Ghimj(index,1186)*K(index,istage,58)- Ghimj(index,1187)*K(index,istage,65)- Ghimj(index,1188)*K(index,istage,66)- Ghimj(index,1189)*K(index,istage,72)- Ghimj(index,1190)*K(index,istage,77)- Ghimj(index,1191)*K(index,istage,82)  - Ghimj(index,1192)*K(index,istage,89)- Ghimj(index,1193)*K(index,istage,91)- Ghimj(index,1194)*K(index,istage,93)- Ghimj(index,1195)*K(index,istage,94)- Ghimj(index,1196)*K(index,istage,98)- Ghimj(index,1197)*K(index,istage,102)- Ghimj(index,1198) *K(index,istage,103)- Ghimj(index,1199)*K(index,istage,104)- Ghimj(index,1200)*K(index,istage,106)- Ghimj(index,1201)*K(index,istage,107)- Ghimj(index,1202)*K(index,istage,108)- Ghimj(index,1203)*K(index,istage,109)- Ghimj(index,1204)*K(index,istage,110) - Ghimj(index,1205)*K(index,istage,113)- Ghimj(index,1206)*K(index,istage,114)- Ghimj(index,1207)*K(index,istage,115)- Ghimj(index,1208)*K(index,istage,117)- Ghimj(index,1209)*K(index,istage,120)- Ghimj(index,1210)*K(index,istage,121) - Ghimj(index,1211)*K(index,istage,122)- Ghimj(index,1212)*K(index,istage,124)- Ghimj(index,1213)*K(index,istage,125)- Ghimj(index,1214)*K(index,istage,126)- Ghimj(index,1215)*K(index,istage,127)- Ghimj(index,1216)*K(index,istage,128) - Ghimj(index,1217)*K(index,istage,129);
        K(index,istage,131) = K(index,istage,131)- Ghimj(index,1227)*K(index,istage,51)- Ghimj(index,1228)*K(index,istage,59)- Ghimj(index,1229)*K(index,istage,75)- Ghimj(index,1230)*K(index,istage,116)- Ghimj(index,1231)*K(index,istage,118)- Ghimj(index,1232)*K(index,istage,120)  - Ghimj(index,1233)*K(index,istage,122)- Ghimj(index,1234)*K(index,istage,123)- Ghimj(index,1235)*K(index,istage,124)- Ghimj(index,1236)*K(index,istage,125)- Ghimj(index,1237)*K(index,istage,126)- Ghimj(index,1238)*K(index,istage,127) - Ghimj(index,1239)*K(index,istage,128)- Ghimj(index,1240)*K(index,istage,129)- Ghimj(index,1241)*K(index,istage,130);
        K(index,istage,132) = K(index,istage,132)- Ghimj(index,1250)*K(index,istage,105)- Ghimj(index,1251)*K(index,istage,114)- Ghimj(index,1252)*K(index,istage,118)- Ghimj(index,1253)*K(index,istage,123)- Ghimj(index,1254)*K(index,istage,124)- Ghimj(index,1255)*K(index,istage,125)  - Ghimj(index,1256)*K(index,istage,126)- Ghimj(index,1257)*K(index,istage,127)- Ghimj(index,1258)*K(index,istage,128)- Ghimj(index,1259)*K(index,istage,129)- Ghimj(index,1260)*K(index,istage,130)- Ghimj(index,1261)*K(index,istage,131);
        K(index,istage,133) = K(index,istage,133)- Ghimj(index,1269)*K(index,istage,59)- Ghimj(index,1270)*K(index,istage,60)- Ghimj(index,1271)*K(index,istage,70)- Ghimj(index,1272)*K(index,istage,76)- Ghimj(index,1273)*K(index,istage,84)- Ghimj(index,1274)*K(index,istage,87)  - Ghimj(index,1275)*K(index,istage,92)- Ghimj(index,1276)*K(index,istage,93)- Ghimj(index,1277)*K(index,istage,94)- Ghimj(index,1278)*K(index,istage,99)- Ghimj(index,1279)*K(index,istage,102)- Ghimj(index,1280)*K(index,istage,109)- Ghimj(index,1281) *K(index,istage,111)- Ghimj(index,1282)*K(index,istage,113)- Ghimj(index,1283)*K(index,istage,115)- Ghimj(index,1284)*K(index,istage,117)- Ghimj(index,1285)*K(index,istage,120)- Ghimj(index,1286)*K(index,istage,121)- Ghimj(index,1287)*K(index,istage,122) - Ghimj(index,1288)*K(index,istage,124)- Ghimj(index,1289)*K(index,istage,125)- Ghimj(index,1290)*K(index,istage,126)- Ghimj(index,1291)*K(index,istage,127)- Ghimj(index,1292)*K(index,istage,128)- Ghimj(index,1293)*K(index,istage,129) - Ghimj(index,1294)*K(index,istage,130)- Ghimj(index,1295)*K(index,istage,131)- Ghimj(index,1296)*K(index,istage,132);
        K(index,istage,134) = K(index,istage,134)- Ghimj(index,1303)*K(index,istage,39)- Ghimj(index,1304)*K(index,istage,41)- Ghimj(index,1305)*K(index,istage,42)- Ghimj(index,1306)*K(index,istage,43)- Ghimj(index,1307)*K(index,istage,51)- Ghimj(index,1308)*K(index,istage,75)  - Ghimj(index,1309)*K(index,istage,112)- Ghimj(index,1310)*K(index,istage,116)- Ghimj(index,1311)*K(index,istage,120)- Ghimj(index,1312)*K(index,istage,122)- Ghimj(index,1313)*K(index,istage,123)- Ghimj(index,1314)*K(index,istage,124) - Ghimj(index,1315)*K(index,istage,125)- Ghimj(index,1316)*K(index,istage,126)- Ghimj(index,1317)*K(index,istage,127)- Ghimj(index,1318)*K(index,istage,128)- Ghimj(index,1319)*K(index,istage,129)- Ghimj(index,1320)*K(index,istage,130) - Ghimj(index,1321)*K(index,istage,131)- Ghimj(index,1322)*K(index,istage,132)- Ghimj(index,1323)*K(index,istage,133);
        K(index,istage,135) = K(index,istage,135)- Ghimj(index,1329)*K(index,istage,0)- Ghimj(index,1330)*K(index,istage,50)- Ghimj(index,1331)*K(index,istage,58)- Ghimj(index,1332)*K(index,istage,59)- Ghimj(index,1333)*K(index,istage,62)- Ghimj(index,1334)*K(index,istage,64)  - Ghimj(index,1335)*K(index,istage,73)- Ghimj(index,1336)*K(index,istage,76)- Ghimj(index,1337)*K(index,istage,77)- Ghimj(index,1338)*K(index,istage,83)- Ghimj(index,1339)*K(index,istage,87)- Ghimj(index,1340)*K(index,istage,91)- Ghimj(index,1341) *K(index,istage,92)- Ghimj(index,1342)*K(index,istage,93)- Ghimj(index,1343)*K(index,istage,94)- Ghimj(index,1344)*K(index,istage,99)- Ghimj(index,1345)*K(index,istage,101)- Ghimj(index,1346)*K(index,istage,102)- Ghimj(index,1347)*K(index,istage,105) - Ghimj(index,1348)*K(index,istage,106)- Ghimj(index,1349)*K(index,istage,109)- Ghimj(index,1350)*K(index,istage,111)- Ghimj(index,1351)*K(index,istage,113)- Ghimj(index,1352)*K(index,istage,114)- Ghimj(index,1353)*K(index,istage,115) - Ghimj(index,1354)*K(index,istage,116)- Ghimj(index,1355)*K(index,istage,117)- Ghimj(index,1356)*K(index,istage,119)- Ghimj(index,1357)*K(index,istage,121)- Ghimj(index,1358)*K(index,istage,123)- Ghimj(index,1359)*K(index,istage,124) - Ghimj(index,1360)*K(index,istage,125)- Ghimj(index,1361)*K(index,istage,126)- Ghimj(index,1362)*K(index,istage,127)- Ghimj(index,1363)*K(index,istage,128)- Ghimj(index,1364)*K(index,istage,129)- Ghimj(index,1365)*K(index,istage,130) - Ghimj(index,1366)*K(index,istage,131)- Ghimj(index,1367)*K(index,istage,132)- Ghimj(index,1368)*K(index,istage,133)- Ghimj(index,1369)*K(index,istage,134);
        K(index,istage,136) = K(index,istage,136)- Ghimj(index,1374)*K(index,istage,73)- Ghimj(index,1375)*K(index,istage,83)- Ghimj(index,1376)*K(index,istage,101)- Ghimj(index,1377)*K(index,istage,105)- Ghimj(index,1378)*K(index,istage,106)- Ghimj(index,1379)*K(index,istage,107)  - Ghimj(index,1380)*K(index,istage,114)- Ghimj(index,1381)*K(index,istage,116)- Ghimj(index,1382)*K(index,istage,117)- Ghimj(index,1383)*K(index,istage,119)- Ghimj(index,1384)*K(index,istage,121)- Ghimj(index,1385)*K(index,istage,123) - Ghimj(index,1386)*K(index,istage,124)- Ghimj(index,1387)*K(index,istage,125)- Ghimj(index,1388)*K(index,istage,126)- Ghimj(index,1389)*K(index,istage,127)- Ghimj(index,1390)*K(index,istage,128)- Ghimj(index,1391)*K(index,istage,129) - Ghimj(index,1392)*K(index,istage,130)- Ghimj(index,1393)*K(index,istage,131)- Ghimj(index,1394)*K(index,istage,132)- Ghimj(index,1395)*K(index,istage,133)- Ghimj(index,1396)*K(index,istage,134)- Ghimj(index,1397)*K(index,istage,135);
        K(index,istage,137) = K(index,istage,137)- Ghimj(index,1401)*K(index,istage,46)- Ghimj(index,1402)*K(index,istage,56)- Ghimj(index,1403)*K(index,istage,62)- Ghimj(index,1404)*K(index,istage,65)- Ghimj(index,1405)*K(index,istage,66)- Ghimj(index,1406)*K(index,istage,69)  - Ghimj(index,1407)*K(index,istage,71)- Ghimj(index,1408)*K(index,istage,73)- Ghimj(index,1409)*K(index,istage,78)- Ghimj(index,1410)*K(index,istage,79)- Ghimj(index,1411)*K(index,istage,81)- Ghimj(index,1412)*K(index,istage,82)- Ghimj(index,1413) *K(index,istage,87)- Ghimj(index,1414)*K(index,istage,88)- Ghimj(index,1415)*K(index,istage,89)- Ghimj(index,1416)*K(index,istage,91)- Ghimj(index,1417)*K(index,istage,92)- Ghimj(index,1418)*K(index,istage,93)- Ghimj(index,1419)*K(index,istage,94) - Ghimj(index,1420)*K(index,istage,96)- Ghimj(index,1421)*K(index,istage,99)- Ghimj(index,1422)*K(index,istage,102)- Ghimj(index,1423)*K(index,istage,103)- Ghimj(index,1424)*K(index,istage,104)- Ghimj(index,1425)*K(index,istage,106) - Ghimj(index,1426)*K(index,istage,107)- Ghimj(index,1427)*K(index,istage,108)- Ghimj(index,1428)*K(index,istage,109)- Ghimj(index,1429)*K(index,istage,110)- Ghimj(index,1430)*K(index,istage,111)- Ghimj(index,1431)*K(index,istage,113) - Ghimj(index,1432)*K(index,istage,114)- Ghimj(index,1433)*K(index,istage,115)- Ghimj(index,1434)*K(index,istage,117)- Ghimj(index,1435)*K(index,istage,119)- Ghimj(index,1436)*K(index,istage,121)- Ghimj(index,1437)*K(index,istage,122) - Ghimj(index,1438)*K(index,istage,124)- Ghimj(index,1439)*K(index,istage,125)- Ghimj(index,1440)*K(index,istage,126)- Ghimj(index,1441)*K(index,istage,127)- Ghimj(index,1442)*K(index,istage,128)- Ghimj(index,1443)*K(index,istage,129) - Ghimj(index,1444)*K(index,istage,130)- Ghimj(index,1445)*K(index,istage,131)- Ghimj(index,1446)*K(index,istage,132)- Ghimj(index,1447)*K(index,istage,133)- Ghimj(index,1448)*K(index,istage,134)- Ghimj(index,1449)*K(index,istage,135) - Ghimj(index,1450)*K(index,istage,136);
        K(index,istage,138) = K(index,istage,138)- Ghimj(index,1453)*K(index,istage,83)- Ghimj(index,1454)*K(index,istage,88)- Ghimj(index,1455)*K(index,istage,97)- Ghimj(index,1456)*K(index,istage,98)- Ghimj(index,1457)*K(index,istage,103)- Ghimj(index,1458)*K(index,istage,104)  - Ghimj(index,1459)*K(index,istage,105)- Ghimj(index,1460)*K(index,istage,106)- Ghimj(index,1461)*K(index,istage,107)- Ghimj(index,1462)*K(index,istage,112)- Ghimj(index,1463)*K(index,istage,114)- Ghimj(index,1464)*K(index,istage,116) - Ghimj(index,1465)*K(index,istage,118)- Ghimj(index,1466)*K(index,istage,119)- Ghimj(index,1467)*K(index,istage,120)- Ghimj(index,1468)*K(index,istage,121)- Ghimj(index,1469)*K(index,istage,122)- Ghimj(index,1470)*K(index,istage,123) - Ghimj(index,1471)*K(index,istage,124)- Ghimj(index,1472)*K(index,istage,125)- Ghimj(index,1473)*K(index,istage,126)- Ghimj(index,1474)*K(index,istage,127)- Ghimj(index,1475)*K(index,istage,128)- Ghimj(index,1476)*K(index,istage,129) - Ghimj(index,1477)*K(index,istage,130)- Ghimj(index,1478)*K(index,istage,131)- Ghimj(index,1479)*K(index,istage,132)- Ghimj(index,1480)*K(index,istage,133)- Ghimj(index,1481)*K(index,istage,134)- Ghimj(index,1482)*K(index,istage,135) - Ghimj(index,1483)*K(index,istage,136)- Ghimj(index,1484)*K(index,istage,137);
        K(index,istage,138) = K(index,istage,138)/ Ghimj(index,1485);
        K(index,istage,137) = (K(index,istage,137)- Ghimj(index,1452)*K(index,istage,138))/(Ghimj(index,1451));
        K(index,istage,136) = (K(index,istage,136)- Ghimj(index,1399)*K(index,istage,137)- Ghimj(index,1400)*K(index,istage,138))/(Ghimj(index,1398));
        K(index,istage,135) = (K(index,istage,135)- Ghimj(index,1371)*K(index,istage,136)- Ghimj(index,1372)*K(index,istage,137)- Ghimj(index,1373)*K(index,istage,138))/(Ghimj(index,1370));
        K(index,istage,134) = (K(index,istage,134)- Ghimj(index,1325)*K(index,istage,135)- Ghimj(index,1326)*K(index,istage,136)- Ghimj(index,1327)*K(index,istage,137)- Ghimj(index,1328)*K(index,istage,138))/(Ghimj(index,1324));
        K(index,istage,133) = (K(index,istage,133)- Ghimj(index,1298)*K(index,istage,134)- Ghimj(index,1299)*K(index,istage,135)- Ghimj(index,1300)*K(index,istage,136)- Ghimj(index,1301)*K(index,istage,137)- Ghimj(index,1302)*K(index,istage,138))/(Ghimj(index,1297));
        K(index,istage,132) = (K(index,istage,132)- Ghimj(index,1263)*K(index,istage,133)- Ghimj(index,1264)*K(index,istage,134)- Ghimj(index,1265)*K(index,istage,135)- Ghimj(index,1266)*K(index,istage,136)- Ghimj(index,1267)*K(index,istage,137)- Ghimj(index,1268)  *K(index,istage,138))/(Ghimj(index,1262));
        K(index,istage,131) = (K(index,istage,131)- Ghimj(index,1243)*K(index,istage,132)- Ghimj(index,1244)*K(index,istage,133)- Ghimj(index,1245)*K(index,istage,134)- Ghimj(index,1246)*K(index,istage,135)- Ghimj(index,1247)*K(index,istage,136)- Ghimj(index,1248)*K(index,istage,137)  - Ghimj(index,1249)*K(index,istage,138))/(Ghimj(index,1242));
        K(index,istage,130) = (K(index,istage,130)- Ghimj(index,1219)*K(index,istage,131)- Ghimj(index,1220)*K(index,istage,132)- Ghimj(index,1221)*K(index,istage,133)- Ghimj(index,1222)*K(index,istage,134)- Ghimj(index,1223)*K(index,istage,135)- Ghimj(index,1224)*K(index,istage,136)  - Ghimj(index,1225)*K(index,istage,137)- Ghimj(index,1226)*K(index,istage,138))/(Ghimj(index,1218));
        K(index,istage,129) = (K(index,istage,129)- Ghimj(index,1177)*K(index,istage,130)- Ghimj(index,1178)*K(index,istage,131)- Ghimj(index,1179)*K(index,istage,132)- Ghimj(index,1180)*K(index,istage,133)- Ghimj(index,1181)*K(index,istage,134)- Ghimj(index,1182)*K(index,istage,135)  - Ghimj(index,1183)*K(index,istage,136)- Ghimj(index,1184)*K(index,istage,137)- Ghimj(index,1185)*K(index,istage,138))/(Ghimj(index,1176));
        K(index,istage,128) = (K(index,istage,128)- Ghimj(index,1139)*K(index,istage,129)- Ghimj(index,1140)*K(index,istage,130)- Ghimj(index,1141)*K(index,istage,131)- Ghimj(index,1142)*K(index,istage,132)- Ghimj(index,1143)*K(index,istage,133)- Ghimj(index,1144)*K(index,istage,134)  - Ghimj(index,1145)*K(index,istage,135)- Ghimj(index,1146)*K(index,istage,136)- Ghimj(index,1147)*K(index,istage,137)- Ghimj(index,1148)*K(index,istage,138))/(Ghimj(index,1138));
        K(index,istage,127) = (K(index,istage,127)- Ghimj(index,1072)*K(index,istage,128)- Ghimj(index,1073)*K(index,istage,129)- Ghimj(index,1074)*K(index,istage,130)- Ghimj(index,1075)*K(index,istage,131)- Ghimj(index,1076)*K(index,istage,132)- Ghimj(index,1077)*K(index,istage,133)  - Ghimj(index,1078)*K(index,istage,134)- Ghimj(index,1079)*K(index,istage,135)- Ghimj(index,1080)*K(index,istage,136)- Ghimj(index,1081)*K(index,istage,137)- Ghimj(index,1082)*K(index,istage,138))/(Ghimj(index,1071));
        K(index,istage,126) = (K(index,istage,126)- Ghimj(index,1024)*K(index,istage,127)- Ghimj(index,1025)*K(index,istage,128)- Ghimj(index,1026)*K(index,istage,129)- Ghimj(index,1027)*K(index,istage,130)- Ghimj(index,1028)*K(index,istage,131)- Ghimj(index,1029)*K(index,istage,132)  - Ghimj(index,1030)*K(index,istage,133)- Ghimj(index,1031)*K(index,istage,134)- Ghimj(index,1032)*K(index,istage,135)- Ghimj(index,1033)*K(index,istage,136)- Ghimj(index,1034)*K(index,istage,137)- Ghimj(index,1035)*K(index,istage,138)) /(Ghimj(index,1023));
        K(index,istage,125) = (K(index,istage,125)- Ghimj(index,935)*K(index,istage,126)- Ghimj(index,936)*K(index,istage,127)- Ghimj(index,937)*K(index,istage,128)- Ghimj(index,938)*K(index,istage,129)- Ghimj(index,939)*K(index,istage,130)- Ghimj(index,940)*K(index,istage,131)  - Ghimj(index,941)*K(index,istage,132)- Ghimj(index,942)*K(index,istage,133)- Ghimj(index,943)*K(index,istage,134)- Ghimj(index,944)*K(index,istage,135)- Ghimj(index,945)*K(index,istage,136)- Ghimj(index,946)*K(index,istage,137)- Ghimj(index,947) *K(index,istage,138))/(Ghimj(index,934));
        K(index,istage,124) = (K(index,istage,124)- Ghimj(index,897)*K(index,istage,125)- Ghimj(index,898)*K(index,istage,126)- Ghimj(index,899)*K(index,istage,127)- Ghimj(index,900)*K(index,istage,128)- Ghimj(index,901)*K(index,istage,129)- Ghimj(index,902)*K(index,istage,130)  - Ghimj(index,903)*K(index,istage,131)- Ghimj(index,904)*K(index,istage,132)- Ghimj(index,905)*K(index,istage,133)- Ghimj(index,906)*K(index,istage,135)- Ghimj(index,907)*K(index,istage,136)- Ghimj(index,908)*K(index,istage,137)- Ghimj(index,909) *K(index,istage,138))/(Ghimj(index,896));
        K(index,istage,123) = (K(index,istage,123)- Ghimj(index,870)*K(index,istage,124)- Ghimj(index,871)*K(index,istage,125)- Ghimj(index,872)*K(index,istage,126)- Ghimj(index,873)*K(index,istage,127)- Ghimj(index,874)*K(index,istage,128)- Ghimj(index,875)*K(index,istage,129)  - Ghimj(index,876)*K(index,istage,130)- Ghimj(index,877)*K(index,istage,131)- Ghimj(index,878)*K(index,istage,132)- Ghimj(index,879)*K(index,istage,133)- Ghimj(index,880)*K(index,istage,134)- Ghimj(index,881)*K(index,istage,135)- Ghimj(index,882) *K(index,istage,136)- Ghimj(index,883)*K(index,istage,137)- Ghimj(index,884)*K(index,istage,138))/(Ghimj(index,869));
        K(index,istage,122) = (K(index,istage,122)- Ghimj(index,848)*K(index,istage,124)- Ghimj(index,849)*K(index,istage,125)- Ghimj(index,850)*K(index,istage,126)- Ghimj(index,851)*K(index,istage,127)- Ghimj(index,852)*K(index,istage,128)- Ghimj(index,853)*K(index,istage,129)  - Ghimj(index,854)*K(index,istage,130)- Ghimj(index,855)*K(index,istage,131)- Ghimj(index,856)*K(index,istage,133)- Ghimj(index,857)*K(index,istage,135)- Ghimj(index,858)*K(index,istage,136)- Ghimj(index,859)*K(index,istage,137)- Ghimj(index,860) *K(index,istage,138))/(Ghimj(index,847));
        K(index,istage,121) = (K(index,istage,121)- Ghimj(index,822)*K(index,istage,124)- Ghimj(index,823)*K(index,istage,125)- Ghimj(index,824)*K(index,istage,126)- Ghimj(index,825)*K(index,istage,127)- Ghimj(index,826)*K(index,istage,129)- Ghimj(index,827)*K(index,istage,133)  - Ghimj(index,828)*K(index,istage,135)- Ghimj(index,829)*K(index,istage,136)- Ghimj(index,830)*K(index,istage,137))/(Ghimj(index,821));
        K(index,istage,120) = (K(index,istage,120)- Ghimj(index,788)*K(index,istage,122)- Ghimj(index,789)*K(index,istage,124)- Ghimj(index,790)*K(index,istage,126)- Ghimj(index,791)*K(index,istage,127)- Ghimj(index,792)*K(index,istage,128)- Ghimj(index,793)*K(index,istage,130)  - Ghimj(index,794)*K(index,istage,133)- Ghimj(index,795)*K(index,istage,135)- Ghimj(index,796)*K(index,istage,136)- Ghimj(index,797)*K(index,istage,137))/(Ghimj(index,787));
        K(index,istage,119) = (K(index,istage,119)- Ghimj(index,768)*K(index,istage,121)- Ghimj(index,769)*K(index,istage,124)- Ghimj(index,770)*K(index,istage,125)- Ghimj(index,771)*K(index,istage,126)- Ghimj(index,772)*K(index,istage,127)- Ghimj(index,773)*K(index,istage,129)  - Ghimj(index,774)*K(index,istage,133)- Ghimj(index,775)*K(index,istage,136)- Ghimj(index,776)*K(index,istage,137))/(Ghimj(index,767));
        K(index,istage,118) = (K(index,istage,118)- Ghimj(index,746)*K(index,istage,123)- Ghimj(index,747)*K(index,istage,125)- Ghimj(index,748)*K(index,istage,126)- Ghimj(index,749)*K(index,istage,127)- Ghimj(index,750)*K(index,istage,128)- Ghimj(index,751)*K(index,istage,129)  - Ghimj(index,752)*K(index,istage,131)- Ghimj(index,753)*K(index,istage,132)- Ghimj(index,754)*K(index,istage,134)- Ghimj(index,755)*K(index,istage,135)- Ghimj(index,756)*K(index,istage,137)- Ghimj(index,757)*K(index,istage,138))/(Ghimj(index,745));
        K(index,istage,117) = (K(index,istage,117)- Ghimj(index,732)*K(index,istage,121)- Ghimj(index,733)*K(index,istage,124)- Ghimj(index,734)*K(index,istage,125)- Ghimj(index,735)*K(index,istage,126)- Ghimj(index,736)*K(index,istage,127)- Ghimj(index,737)*K(index,istage,129)  - Ghimj(index,738)*K(index,istage,133)- Ghimj(index,739)*K(index,istage,136)- Ghimj(index,740)*K(index,istage,137))/(Ghimj(index,731));
        K(index,istage,116) = (K(index,istage,116)- Ghimj(index,715)*K(index,istage,123)- Ghimj(index,716)*K(index,istage,127)- Ghimj(index,717)*K(index,istage,128)- Ghimj(index,718)*K(index,istage,131)- Ghimj(index,719)*K(index,istage,134)- Ghimj(index,720)*K(index,istage,135)  - Ghimj(index,721)*K(index,istage,138))/(Ghimj(index,714));
        K(index,istage,115) = (K(index,istage,115)- Ghimj(index,707)*K(index,istage,124)- Ghimj(index,708)*K(index,istage,126)- Ghimj(index,709)*K(index,istage,127)- Ghimj(index,710)*K(index,istage,129)- Ghimj(index,711)*K(index,istage,133)- Ghimj(index,712)*K(index,istage,136)  - Ghimj(index,713)*K(index,istage,137))/(Ghimj(index,706));
        K(index,istage,114) = (K(index,istage,114)- Ghimj(index,698)*K(index,istage,126)- Ghimj(index,699)*K(index,istage,127)- Ghimj(index,700)*K(index,istage,129)- Ghimj(index,701)*K(index,istage,132)- Ghimj(index,702)*K(index,istage,136))/(Ghimj(index,697));
        K(index,istage,113) = (K(index,istage,113)- Ghimj(index,690)*K(index,istage,124)- Ghimj(index,691)*K(index,istage,125)- Ghimj(index,692)*K(index,istage,126)- Ghimj(index,693)*K(index,istage,133)- Ghimj(index,694)*K(index,istage,135)- Ghimj(index,695)*K(index,istage,136)  - Ghimj(index,696)*K(index,istage,137))/(Ghimj(index,689));
        K(index,istage,112) = (K(index,istage,112)- Ghimj(index,678)*K(index,istage,116)- Ghimj(index,679)*K(index,istage,123)- Ghimj(index,680)*K(index,istage,126)- Ghimj(index,681)*K(index,istage,128)- Ghimj(index,682)*K(index,istage,134)- Ghimj(index,683)*K(index,istage,137)  - Ghimj(index,684)*K(index,istage,138))/(Ghimj(index,677));
        K(index,istage,111) = (K(index,istage,111)- Ghimj(index,670)*K(index,istage,115)- Ghimj(index,671)*K(index,istage,124)- Ghimj(index,672)*K(index,istage,125)- Ghimj(index,673)*K(index,istage,126)- Ghimj(index,674)*K(index,istage,133)- Ghimj(index,675)*K(index,istage,136)  - Ghimj(index,676)*K(index,istage,137))/(Ghimj(index,669));
        K(index,istage,110) = (K(index,istage,110)- Ghimj(index,660)*K(index,istage,124)- Ghimj(index,661)*K(index,istage,125)- Ghimj(index,662)*K(index,istage,126)- Ghimj(index,663)*K(index,istage,133)- Ghimj(index,664)*K(index,istage,136)- Ghimj(index,665)*K(index,istage,137))  /(Ghimj(index,659));
        K(index,istage,109) = (K(index,istage,109)- Ghimj(index,649)*K(index,istage,124)- Ghimj(index,650)*K(index,istage,125)- Ghimj(index,651)*K(index,istage,126)- Ghimj(index,652)*K(index,istage,133)- Ghimj(index,653)*K(index,istage,136)- Ghimj(index,654)*K(index,istage,137))  /(Ghimj(index,648));
        K(index,istage,108) = (K(index,istage,108)- Ghimj(index,637)*K(index,istage,109)- Ghimj(index,638)*K(index,istage,113)- Ghimj(index,639)*K(index,istage,115)- Ghimj(index,640)*K(index,istage,124)- Ghimj(index,641)*K(index,istage,125)- Ghimj(index,642)*K(index,istage,126)  - Ghimj(index,643)*K(index,istage,133)- Ghimj(index,644)*K(index,istage,135)- Ghimj(index,645)*K(index,istage,136)- Ghimj(index,646)*K(index,istage,137))/(Ghimj(index,636));
        K(index,istage,107) = (K(index,istage,107)- Ghimj(index,627)*K(index,istage,124)- Ghimj(index,628)*K(index,istage,126)- Ghimj(index,629)*K(index,istage,136))/(Ghimj(index,626));
        K(index,istage,106) = (K(index,istage,106)- Ghimj(index,623)*K(index,istage,124)- Ghimj(index,624)*K(index,istage,126)- Ghimj(index,625)*K(index,istage,136))/(Ghimj(index,622));
        K(index,istage,105) = (K(index,istage,105)- Ghimj(index,617)*K(index,istage,128)- Ghimj(index,618)*K(index,istage,129)- Ghimj(index,619)*K(index,istage,132)- Ghimj(index,620)*K(index,istage,135)- Ghimj(index,621)*K(index,istage,138))/(Ghimj(index,616));
        K(index,istage,104) = (K(index,istage,104)- Ghimj(index,611)*K(index,istage,125)- Ghimj(index,612)*K(index,istage,126)- Ghimj(index,613)*K(index,istage,127)- Ghimj(index,614)*K(index,istage,129)- Ghimj(index,615)*K(index,istage,137))/(Ghimj(index,610));
        K(index,istage,103) = (K(index,istage,103)- Ghimj(index,606)*K(index,istage,124)- Ghimj(index,607)*K(index,istage,126)- Ghimj(index,608)*K(index,istage,127)- Ghimj(index,609)*K(index,istage,129))/(Ghimj(index,605));
        K(index,istage,102) = (K(index,istage,102)- Ghimj(index,601)*K(index,istage,125)- Ghimj(index,602)*K(index,istage,126)- Ghimj(index,603)*K(index,istage,133)- Ghimj(index,604)*K(index,istage,137))/(Ghimj(index,600));
        K(index,istage,101) = (K(index,istage,101)- Ghimj(index,587)*K(index,istage,105)- Ghimj(index,588)*K(index,istage,114)- Ghimj(index,589)*K(index,istage,116)- Ghimj(index,590)*K(index,istage,119)- Ghimj(index,591)*K(index,istage,123)- Ghimj(index,592)*K(index,istage,126)  - Ghimj(index,593)*K(index,istage,128)- Ghimj(index,594)*K(index,istage,130)- Ghimj(index,595)*K(index,istage,135)- Ghimj(index,596)*K(index,istage,136)- Ghimj(index,597)*K(index,istage,138))/(Ghimj(index,586));
        K(index,istage,100) = (K(index,istage,100)- Ghimj(index,574)*K(index,istage,105)- Ghimj(index,575)*K(index,istage,112)- Ghimj(index,576)*K(index,istage,116)- Ghimj(index,577)*K(index,istage,118)- Ghimj(index,578)*K(index,istage,123)- Ghimj(index,579)*K(index,istage,126)  - Ghimj(index,580)*K(index,istage,127)- Ghimj(index,581)*K(index,istage,129)- Ghimj(index,582)*K(index,istage,132)- Ghimj(index,583)*K(index,istage,134)- Ghimj(index,584)*K(index,istage,138))/(Ghimj(index,573));
        K(index,istage,99) = (K(index,istage,99)- Ghimj(index,566)*K(index,istage,102)- Ghimj(index,567)*K(index,istage,111)- Ghimj(index,568)*K(index,istage,125)- Ghimj(index,569)*K(index,istage,126)- Ghimj(index,570)*K(index,istage,133)- Ghimj(index,571)*K(index,istage,137))  /(Ghimj(index,565));
        K(index,istage,98) = (K(index,istage,98)- Ghimj(index,558)*K(index,istage,107)- Ghimj(index,559)*K(index,istage,120)- Ghimj(index,560)*K(index,istage,124)- Ghimj(index,561)*K(index,istage,126)- Ghimj(index,562)*K(index,istage,127))/(Ghimj(index,557));
        K(index,istage,97) = (K(index,istage,97)- Ghimj(index,550)*K(index,istage,98)- Ghimj(index,551)*K(index,istage,120)- Ghimj(index,552)*K(index,istage,122)- Ghimj(index,553)*K(index,istage,126)- Ghimj(index,554)*K(index,istage,127)- Ghimj(index,555)*K(index,istage,130)- Ghimj(index,556)  *K(index,istage,137))/(Ghimj(index,549));
        K(index,istage,96) = (K(index,istage,96)- Ghimj(index,539)*K(index,istage,107)- Ghimj(index,540)*K(index,istage,108)- Ghimj(index,541)*K(index,istage,109)- Ghimj(index,542)*K(index,istage,110)- Ghimj(index,543)*K(index,istage,113)- Ghimj(index,544)*K(index,istage,124)  - Ghimj(index,545)*K(index,istage,125)- Ghimj(index,546)*K(index,istage,126)- Ghimj(index,547)*K(index,istage,133)- Ghimj(index,548)*K(index,istage,137))/(Ghimj(index,538));
        K(index,istage,95) = (K(index,istage,95)- Ghimj(index,515)*K(index,istage,96)- Ghimj(index,516)*K(index,istage,98)- Ghimj(index,517)*K(index,istage,103)- Ghimj(index,518)*K(index,istage,106)- Ghimj(index,519)*K(index,istage,107)- Ghimj(index,520)*K(index,istage,109)- Ghimj(index,521)  *K(index,istage,110)- Ghimj(index,522)*K(index,istage,113)- Ghimj(index,523)*K(index,istage,119)- Ghimj(index,524)*K(index,istage,121)- Ghimj(index,525)*K(index,istage,124)- Ghimj(index,526)*K(index,istage,125)- Ghimj(index,527)*K(index,istage,126) - Ghimj(index,528)*K(index,istage,127)- Ghimj(index,529)*K(index,istage,129)- Ghimj(index,530)*K(index,istage,130)- Ghimj(index,531)*K(index,istage,133)- Ghimj(index,532)*K(index,istage,135)- Ghimj(index,533)*K(index,istage,136)- Ghimj(index,534) *K(index,istage,137))/(Ghimj(index,514));
        K(index,istage,94) = (K(index,istage,94)- Ghimj(index,506)*K(index,istage,125)- Ghimj(index,507)*K(index,istage,126)- Ghimj(index,508)*K(index,istage,133)- Ghimj(index,509)*K(index,istage,137))/(Ghimj(index,505));
        K(index,istage,93) = (K(index,istage,93)- Ghimj(index,498)*K(index,istage,125)- Ghimj(index,499)*K(index,istage,126)- Ghimj(index,500)*K(index,istage,133)- Ghimj(index,501)*K(index,istage,137))/(Ghimj(index,497));
        K(index,istage,92) = (K(index,istage,92)- Ghimj(index,490)*K(index,istage,124)- Ghimj(index,491)*K(index,istage,126)- Ghimj(index,492)*K(index,istage,133)- Ghimj(index,493)*K(index,istage,135)- Ghimj(index,494)*K(index,istage,137))/(Ghimj(index,489));
        K(index,istage,91) = (K(index,istage,91)- Ghimj(index,482)*K(index,istage,106)- Ghimj(index,483)*K(index,istage,109)- Ghimj(index,484)*K(index,istage,126)- Ghimj(index,485)*K(index,istage,133)- Ghimj(index,486)*K(index,istage,136))/(Ghimj(index,481));
        K(index,istage,90) = (K(index,istage,90)- Ghimj(index,470)*K(index,istage,100)- Ghimj(index,471)*K(index,istage,105)- Ghimj(index,472)*K(index,istage,112)- Ghimj(index,473)*K(index,istage,116)- Ghimj(index,474)*K(index,istage,118)- Ghimj(index,475)*K(index,istage,123)  - Ghimj(index,476)*K(index,istage,127)- Ghimj(index,477)*K(index,istage,129)- Ghimj(index,478)*K(index,istage,132)- Ghimj(index,479)*K(index,istage,134)- Ghimj(index,480)*K(index,istage,138))/(Ghimj(index,469));
        K(index,istage,89) = (K(index,istage,89)- Ghimj(index,458)*K(index,istage,93)- Ghimj(index,459)*K(index,istage,94)- Ghimj(index,460)*K(index,istage,102)- Ghimj(index,461)*K(index,istage,107)- Ghimj(index,462)*K(index,istage,109)- Ghimj(index,463)*K(index,istage,113)- Ghimj(index,464)  *K(index,istage,117)- Ghimj(index,465)*K(index,istage,124)- Ghimj(index,466)*K(index,istage,125)- Ghimj(index,467)*K(index,istage,126))/(Ghimj(index,457));
        K(index,istage,88) = (K(index,istage,88)- Ghimj(index,451)*K(index,istage,103)- Ghimj(index,452)*K(index,istage,106)- Ghimj(index,453)*K(index,istage,124)- Ghimj(index,454)*K(index,istage,126)- Ghimj(index,455)*K(index,istage,127)- Ghimj(index,456)*K(index,istage,137))  /(Ghimj(index,450));
        K(index,istage,87) = (K(index,istage,87)- Ghimj(index,445)*K(index,istage,92)- Ghimj(index,446)*K(index,istage,124)- Ghimj(index,447)*K(index,istage,126)- Ghimj(index,448)*K(index,istage,135)- Ghimj(index,449)*K(index,istage,137))/(Ghimj(index,444));
        K(index,istage,86) = (K(index,istage,86)- Ghimj(index,437)*K(index,istage,93)- Ghimj(index,438)*K(index,istage,125)- Ghimj(index,439)*K(index,istage,126)- Ghimj(index,440)*K(index,istage,133)- Ghimj(index,441)*K(index,istage,137))/(Ghimj(index,436));
        K(index,istage,85) = (K(index,istage,85)- Ghimj(index,428)*K(index,istage,102)- Ghimj(index,429)*K(index,istage,111)- Ghimj(index,430)*K(index,istage,125)- Ghimj(index,431)*K(index,istage,126)- Ghimj(index,432)*K(index,istage,133)- Ghimj(index,433)*K(index,istage,137))  /(Ghimj(index,427));
        K(index,istage,84) = (K(index,istage,84)- Ghimj(index,422)*K(index,istage,92)- Ghimj(index,423)*K(index,istage,124)- Ghimj(index,424)*K(index,istage,135)- Ghimj(index,425)*K(index,istage,137))/(Ghimj(index,421));
        K(index,istage,83) = (K(index,istage,83)- Ghimj(index,417)*K(index,istage,128)- Ghimj(index,418)*K(index,istage,135)- Ghimj(index,419)*K(index,istage,136)- Ghimj(index,420)*K(index,istage,138))/(Ghimj(index,416));
        K(index,istage,82) = (K(index,istage,82)- Ghimj(index,413)*K(index,istage,113)- Ghimj(index,414)*K(index,istage,126)- Ghimj(index,415)*K(index,istage,137))/(Ghimj(index,412));
        K(index,istage,81) = (K(index,istage,81)- Ghimj(index,406)*K(index,istage,114)- Ghimj(index,407)*K(index,istage,124)- Ghimj(index,408)*K(index,istage,126)- Ghimj(index,409)*K(index,istage,127)- Ghimj(index,410)*K(index,istage,129)- Ghimj(index,411)*K(index,istage,136))  /(Ghimj(index,405));
        K(index,istage,80) = (K(index,istage,80)- Ghimj(index,398)*K(index,istage,90)- Ghimj(index,399)*K(index,istage,112)- Ghimj(index,400)*K(index,istage,116)- Ghimj(index,401)*K(index,istage,127)- Ghimj(index,402)*K(index,istage,129)- Ghimj(index,403)*K(index,istage,134)- Ghimj(index,404)  *K(index,istage,138))/(Ghimj(index,397));
        K(index,istage,79) = (K(index,istage,79)- Ghimj(index,394)*K(index,istage,102)- Ghimj(index,395)*K(index,istage,126)- Ghimj(index,396)*K(index,istage,137))/(Ghimj(index,393));
        K(index,istage,78) = (K(index,istage,78)- Ghimj(index,387)*K(index,istage,103)- Ghimj(index,388)*K(index,istage,106)- Ghimj(index,389)*K(index,istage,107)- Ghimj(index,390)*K(index,istage,110)- Ghimj(index,391)*K(index,istage,124)- Ghimj(index,392)*K(index,istage,126))  /(Ghimj(index,386));
        K(index,istage,77) = (K(index,istage,77)- Ghimj(index,383)*K(index,istage,121)- Ghimj(index,384)*K(index,istage,126)- Ghimj(index,385)*K(index,istage,135))/(Ghimj(index,382));
        K(index,istage,76) = (K(index,istage,76)- Ghimj(index,378)*K(index,istage,87)- Ghimj(index,379)*K(index,istage,126)- Ghimj(index,380)*K(index,istage,133)- Ghimj(index,381)*K(index,istage,135))/(Ghimj(index,377));
        K(index,istage,75) = (K(index,istage,75)- Ghimj(index,375)*K(index,istage,120)- Ghimj(index,376)*K(index,istage,126))/(Ghimj(index,374));
        K(index,istage,74) = (K(index,istage,74)- Ghimj(index,369)*K(index,istage,117)- Ghimj(index,370)*K(index,istage,121)- Ghimj(index,371)*K(index,istage,125)- Ghimj(index,372)*K(index,istage,126)- Ghimj(index,373)*K(index,istage,137))/(Ghimj(index,368));
        K(index,istage,73) = (K(index,istage,73)- Ghimj(index,365)*K(index,istage,126)- Ghimj(index,366)*K(index,istage,135)- Ghimj(index,367)*K(index,istage,137))/(Ghimj(index,364));
        K(index,istage,72) = (K(index,istage,72)- Ghimj(index,361)*K(index,istage,94)- Ghimj(index,362)*K(index,istage,126)- Ghimj(index,363)*K(index,istage,137))/(Ghimj(index,360));
        K(index,istage,71) = (K(index,istage,71)- Ghimj(index,357)*K(index,istage,117)- Ghimj(index,358)*K(index,istage,126)- Ghimj(index,359)*K(index,istage,137))/(Ghimj(index,356));
        K(index,istage,70) = (K(index,istage,70)- Ghimj(index,353)*K(index,istage,84)- Ghimj(index,354)*K(index,istage,87)- Ghimj(index,355)*K(index,istage,126))/(Ghimj(index,352));
        K(index,istage,69) = (K(index,istage,69)- Ghimj(index,348)*K(index,istage,93)- Ghimj(index,349)*K(index,istage,126)- Ghimj(index,350)*K(index,istage,137))/(Ghimj(index,347));
        K(index,istage,68) = (K(index,istage,68)- Ghimj(index,344)*K(index,istage,99)- Ghimj(index,345)*K(index,istage,126)- Ghimj(index,346)*K(index,istage,137))/(Ghimj(index,343));
        K(index,istage,67) = (K(index,istage,67)- Ghimj(index,340)*K(index,istage,115)- Ghimj(index,341)*K(index,istage,126)- Ghimj(index,342)*K(index,istage,137))/(Ghimj(index,339));
        K(index,istage,66) = (K(index,istage,66)- Ghimj(index,336)*K(index,istage,109)- Ghimj(index,337)*K(index,istage,126)- Ghimj(index,338)*K(index,istage,137))/(Ghimj(index,335));
        K(index,istage,65) = (K(index,istage,65)- Ghimj(index,332)*K(index,istage,114)- Ghimj(index,333)*K(index,istage,126)- Ghimj(index,334)*K(index,istage,132))/(Ghimj(index,331));
        K(index,istage,64) = (K(index,istage,64)- Ghimj(index,328)*K(index,istage,113)- Ghimj(index,329)*K(index,istage,126)- Ghimj(index,330)*K(index,istage,135))/(Ghimj(index,327));
        K(index,istage,63) = (K(index,istage,63)- Ghimj(index,324)*K(index,istage,121)- Ghimj(index,325)*K(index,istage,126)- Ghimj(index,326)*K(index,istage,137))/(Ghimj(index,323));
        K(index,istage,62) = (K(index,istage,62)- Ghimj(index,320)*K(index,istage,93)- Ghimj(index,321)*K(index,istage,126)- Ghimj(index,322)*K(index,istage,133))/(Ghimj(index,319));
        K(index,istage,61) = (K(index,istage,61)- Ghimj(index,316)*K(index,istage,70)- Ghimj(index,317)*K(index,istage,87)- Ghimj(index,318)*K(index,istage,126))/(Ghimj(index,315));
        K(index,istage,60) = (K(index,istage,60)- Ghimj(index,311)*K(index,istage,92)- Ghimj(index,312)*K(index,istage,120)- Ghimj(index,313)*K(index,istage,133)- Ghimj(index,314)*K(index,istage,135))/(Ghimj(index,310));
        K(index,istage,59) = (K(index,istage,59)- Ghimj(index,307)*K(index,istage,133)- Ghimj(index,308)*K(index,istage,135))/(Ghimj(index,306));
        K(index,istage,58) = (K(index,istage,58)- Ghimj(index,304)*K(index,istage,91)- Ghimj(index,305)*K(index,istage,126))/(Ghimj(index,303));
        K(index,istage,57) = (K(index,istage,57)- Ghimj(index,301)*K(index,istage,120)- Ghimj(index,302)*K(index,istage,126))/(Ghimj(index,300));
        K(index,istage,56) = (K(index,istage,56)- Ghimj(index,297)*K(index,istage,65)- Ghimj(index,298)*K(index,istage,81)- Ghimj(index,299)*K(index,istage,126))/(Ghimj(index,296));
        K(index,istage,55) = (K(index,istage,55)- Ghimj(index,295)*K(index,istage,126))/(Ghimj(index,294));
        K(index,istage,54) = (K(index,istage,54)- Ghimj(index,293)*K(index,istage,126))/(Ghimj(index,292));
        K(index,istage,53) = (K(index,istage,53)- Ghimj(index,291)*K(index,istage,126))/(Ghimj(index,290));
        K(index,istage,52) = (K(index,istage,52)- Ghimj(index,289)*K(index,istage,126))/(Ghimj(index,288));
        K(index,istage,51) = (K(index,istage,51)- Ghimj(index,286)*K(index,istage,132)- Ghimj(index,287)*K(index,istage,134))/(Ghimj(index,285));
        K(index,istage,50) = (K(index,istage,50)- Ghimj(index,283)*K(index,istage,83)- Ghimj(index,284)*K(index,istage,138))/(Ghimj(index,282));
        K(index,istage,49) = (K(index,istage,49)- Ghimj(index,281)*K(index,istage,126))/(Ghimj(index,280));
        K(index,istage,48) = (K(index,istage,48)- Ghimj(index,279)*K(index,istage,126))/(Ghimj(index,278));
        K(index,istage,47) = (K(index,istage,47)- Ghimj(index,277)*K(index,istage,126))/(Ghimj(index,276));
        K(index,istage,46) = (K(index,istage,46)- Ghimj(index,273)*K(index,istage,81)- Ghimj(index,274)*K(index,istage,124)- Ghimj(index,275)*K(index,istage,137))/(Ghimj(index,272));
        K(index,istage,45) = (K(index,istage,45)- Ghimj(index,271)*K(index,istage,126))/(Ghimj(index,270));
        K(index,istage,44) = (K(index,istage,44)- Ghimj(index,269)*K(index,istage,126))/(Ghimj(index,268));
        K(index,istage,43) = (K(index,istage,43)- Ghimj(index,267)*K(index,istage,120))/(Ghimj(index,266));
        K(index,istage,42) = (K(index,istage,42)- Ghimj(index,265)*K(index,istage,120))/(Ghimj(index,264));
        K(index,istage,41) = (K(index,istage,41)- Ghimj(index,263)*K(index,istage,120))/(Ghimj(index,262));
        K(index,istage,40) = (K(index,istage,40)- Ghimj(index,261)*K(index,istage,126))/(Ghimj(index,260));
        K(index,istage,39) = (K(index,istage,39)- Ghimj(index,259)*K(index,istage,134))/(Ghimj(index,258));
        K(index,istage,38) = (K(index,istage,38)- Ghimj(index,256)*K(index,istage,68)- Ghimj(index,257)*K(index,istage,126))/(Ghimj(index,255));
        K(index,istage,37) = (K(index,istage,37)- Ghimj(index,252)*K(index,istage,52)- Ghimj(index,253)*K(index,istage,54)- Ghimj(index,254)*K(index,istage,55))/(Ghimj(index,251));
        K(index,istage,36) = (K(index,istage,36)- Ghimj(index,245)*K(index,istage,44)- Ghimj(index,246)*K(index,istage,45)- Ghimj(index,247)*K(index,istage,52)- Ghimj(index,248)*K(index,istage,54)- Ghimj(index,249)*K(index,istage,55)- Ghimj(index,250)*K(index,istage,126))/(Ghimj(index,244));
        K(index,istage,35) = (K(index,istage,35)- Ghimj(index,234)*K(index,istage,93)- Ghimj(index,235)*K(index,istage,94)- Ghimj(index,236)*K(index,istage,99)- Ghimj(index,237)*K(index,istage,102)- Ghimj(index,238)*K(index,istage,109)- Ghimj(index,239)*K(index,istage,113)- Ghimj(index,240)  *K(index,istage,115)- Ghimj(index,241)*K(index,istage,117)- Ghimj(index,242)*K(index,istage,121)- Ghimj(index,243)*K(index,istage,133))/(Ghimj(index,233));
        K(index,istage,34) = (K(index,istage,34)- Ghimj(index,207)*K(index,istage,50)- Ghimj(index,208)*K(index,istage,51)- Ghimj(index,209)*K(index,istage,59)- Ghimj(index,210)*K(index,istage,60)- Ghimj(index,211)*K(index,istage,65)- Ghimj(index,212)*K(index,istage,73)- Ghimj(index,213)  *K(index,istage,76)- Ghimj(index,214)*K(index,istage,93)- Ghimj(index,215)*K(index,istage,94)- Ghimj(index,216)*K(index,istage,99)- Ghimj(index,217)*K(index,istage,100)- Ghimj(index,218)*K(index,istage,101)- Ghimj(index,219)*K(index,istage,102)- Ghimj(index,220) *K(index,istage,109)- Ghimj(index,221)*K(index,istage,113)- Ghimj(index,222)*K(index,istage,114)- Ghimj(index,223)*K(index,istage,115)- Ghimj(index,224)*K(index,istage,117)- Ghimj(index,225)*K(index,istage,121)- Ghimj(index,226)*K(index,istage,122) - Ghimj(index,227)*K(index,istage,125)- Ghimj(index,228)*K(index,istage,126)- Ghimj(index,229)*K(index,istage,127)- Ghimj(index,230)*K(index,istage,129)- Ghimj(index,231)*K(index,istage,133)- Ghimj(index,232)*K(index,istage,137))/(Ghimj(index,206));
        K(index,istage,33) = (K(index,istage,33)- Ghimj(index,203)*K(index,istage,125)- Ghimj(index,204)*K(index,istage,133))/(Ghimj(index,202));
        K(index,istage,32) = (K(index,istage,32)- Ghimj(index,195)*K(index,istage,41)- Ghimj(index,196)*K(index,istage,42)- Ghimj(index,197)*K(index,istage,43)- Ghimj(index,198)*K(index,istage,57)- Ghimj(index,199)*K(index,istage,75)- Ghimj(index,200)*K(index,istage,120)- Ghimj(index,201)  *K(index,istage,126))/(Ghimj(index,194));
        K(index,istage,31) = (K(index,istage,31)- Ghimj(index,191)*K(index,istage,53)- Ghimj(index,192)*K(index,istage,126))/(Ghimj(index,190));
        K(index,istage,30) = (K(index,istage,30)- Ghimj(index,186)*K(index,istage,133)- Ghimj(index,187)*K(index,istage,137))/(Ghimj(index,185));
        K(index,istage,29) = (K(index,istage,29)- Ghimj(index,183)*K(index,istage,124)- Ghimj(index,184)*K(index,istage,126))/(Ghimj(index,182));
        K(index,istage,28) = (K(index,istage,28)- Ghimj(index,171)*K(index,istage,103)- Ghimj(index,172)*K(index,istage,106)- Ghimj(index,173)*K(index,istage,107)- Ghimj(index,174)*K(index,istage,110)- Ghimj(index,175)*K(index,istage,117)- Ghimj(index,176)*K(index,istage,119)  - Ghimj(index,177)*K(index,istage,121)- Ghimj(index,178)*K(index,istage,124)- Ghimj(index,179)*K(index,istage,125)- Ghimj(index,180)*K(index,istage,130)- Ghimj(index,181)*K(index,istage,136))/(Ghimj(index,170));
        K(index,istage,27) = (K(index,istage,27)- Ghimj(index,164)*K(index,istage,60)- Ghimj(index,165)*K(index,istage,98)- Ghimj(index,166)*K(index,istage,120)- Ghimj(index,167)*K(index,istage,124)- Ghimj(index,168)*K(index,istage,128)- Ghimj(index,169)*K(index,istage,131))  /(Ghimj(index,163));
        K(index,istage,26) = (K(index,istage,26)- Ghimj(index,149)*K(index,istage,83)- Ghimj(index,150)*K(index,istage,84)- Ghimj(index,151)*K(index,istage,87)- Ghimj(index,152)*K(index,istage,92)- Ghimj(index,153)*K(index,istage,105)- Ghimj(index,154)*K(index,istage,116)- Ghimj(index,155)  *K(index,istage,123)- Ghimj(index,156)*K(index,istage,124)- Ghimj(index,157)*K(index,istage,128)- Ghimj(index,158)*K(index,istage,131)- Ghimj(index,159)*K(index,istage,135)- Ghimj(index,160)*K(index,istage,136)- Ghimj(index,161)*K(index,istage,137) - Ghimj(index,162)*K(index,istage,138))/(Ghimj(index,148));
        K(index,istage,25) = (K(index,istage,25)- Ghimj(index,141)*K(index,istage,97)- Ghimj(index,142)*K(index,istage,120)- Ghimj(index,143)*K(index,istage,122)- Ghimj(index,144)*K(index,istage,124)- Ghimj(index,145)*K(index,istage,126)- Ghimj(index,146)*K(index,istage,131)- Ghimj(index,147)  *K(index,istage,137))/(Ghimj(index,140));
        K(index,istage,24) = (K(index,istage,24)- Ghimj(index,124)*K(index,istage,39)- Ghimj(index,125)*K(index,istage,57)- Ghimj(index,126)*K(index,istage,75)- Ghimj(index,127)*K(index,istage,83)- Ghimj(index,128)*K(index,istage,105)- Ghimj(index,129)*K(index,istage,112)- Ghimj(index,130)  *K(index,istage,116)- Ghimj(index,131)*K(index,istage,118)- Ghimj(index,132)*K(index,istage,120)- Ghimj(index,133)*K(index,istage,123)- Ghimj(index,134)*K(index,istage,125)- Ghimj(index,135)*K(index,istage,126)- Ghimj(index,136)*K(index,istage,131) - Ghimj(index,137)*K(index,istage,132)- Ghimj(index,138)*K(index,istage,134)- Ghimj(index,139)*K(index,istage,138))/(Ghimj(index,123));
        K(index,istage,23) = (K(index,istage,23)- Ghimj(index,113)*K(index,istage,105)- Ghimj(index,114)*K(index,istage,112)- Ghimj(index,115)*K(index,istage,116)- Ghimj(index,116)*K(index,istage,118)- Ghimj(index,117)*K(index,istage,123)- Ghimj(index,118)*K(index,istage,125)  - Ghimj(index,119)*K(index,istage,131)- Ghimj(index,120)*K(index,istage,132)- Ghimj(index,121)*K(index,istage,134)- Ghimj(index,122)*K(index,istage,138))/(Ghimj(index,112));
        K(index,istage,22) = (K(index,istage,22)- Ghimj(index,76)*K(index,istage,39)- Ghimj(index,77)*K(index,istage,57)- Ghimj(index,78)*K(index,istage,60)- Ghimj(index,79)*K(index,istage,75)- Ghimj(index,80)*K(index,istage,83)- Ghimj(index,81)*K(index,istage,84)- Ghimj(index,82)*K(index,istage,87)  - Ghimj(index,83)*K(index,istage,92)- Ghimj(index,84)*K(index,istage,97)- Ghimj(index,85)*K(index,istage,98)- Ghimj(index,86)*K(index,istage,103)- Ghimj(index,87)*K(index,istage,105)- Ghimj(index,88)*K(index,istage,106)- Ghimj(index,89)*K(index,istage,107)- Ghimj(index,90) *K(index,istage,110)- Ghimj(index,91)*K(index,istage,112)- Ghimj(index,92)*K(index,istage,116)- Ghimj(index,93)*K(index,istage,117)- Ghimj(index,94)*K(index,istage,118)- Ghimj(index,95)*K(index,istage,119)- Ghimj(index,96)*K(index,istage,120)- Ghimj(index,97) *K(index,istage,121)- Ghimj(index,98)*K(index,istage,122)- Ghimj(index,99)*K(index,istage,123)- Ghimj(index,100)*K(index,istage,124)- Ghimj(index,101)*K(index,istage,125)- Ghimj(index,102)*K(index,istage,126)- Ghimj(index,103)*K(index,istage,128)- Ghimj(index,104) *K(index,istage,130)- Ghimj(index,105)*K(index,istage,131)- Ghimj(index,106)*K(index,istage,132)- Ghimj(index,107)*K(index,istage,134)- Ghimj(index,108)*K(index,istage,135)- Ghimj(index,109)*K(index,istage,136)- Ghimj(index,110)*K(index,istage,137) - Ghimj(index,111)*K(index,istage,138))/(Ghimj(index,75));
        K(index,istage,21) = (K(index,istage,21)- Ghimj(index,73)*K(index,istage,120)- Ghimj(index,74)*K(index,istage,128))/(Ghimj(index,72));
        K(index,istage,20) = (K(index,istage,20)- Ghimj(index,70)*K(index,istage,124)- Ghimj(index,71)*K(index,istage,137))/(Ghimj(index,69));
        K(index,istage,19) = K(index,istage,19)/ Ghimj(index,68);
        K(index,istage,18) = (K(index,istage,18)- Ghimj(index,65)*K(index,istage,120)- Ghimj(index,66)*K(index,istage,126))/(Ghimj(index,64));
        K(index,istage,17) = (K(index,istage,17)- Ghimj(index,63)*K(index,istage,120))/(Ghimj(index,62));
        K(index,istage,16) = (K(index,istage,16)- Ghimj(index,61)*K(index,istage,120))/(Ghimj(index,60));
        K(index,istage,15) = (K(index,istage,15)- Ghimj(index,59)*K(index,istage,120))/(Ghimj(index,58));
        K(index,istage,14) = (K(index,istage,14)- Ghimj(index,53)*K(index,istage,15)- Ghimj(index,54)*K(index,istage,16)- Ghimj(index,55)*K(index,istage,17)- Ghimj(index,56)*K(index,istage,18)- Ghimj(index,57)*K(index,istage,120))/(Ghimj(index,52));
        K(index,istage,13) = (K(index,istage,13)- Ghimj(index,49)*K(index,istage,83))/(Ghimj(index,48));
        K(index,istage,12) = (K(index,istage,12)- Ghimj(index,47)*K(index,istage,83))/(Ghimj(index,46));
        K(index,istage,11) = (K(index,istage,11)- Ghimj(index,44)*K(index,istage,56)- Ghimj(index,45)*K(index,istage,126))/(Ghimj(index,43));
        K(index,istage,10) = (K(index,istage,10)- Ghimj(index,39)*K(index,istage,46)- Ghimj(index,40)*K(index,istage,65)- Ghimj(index,41)*K(index,istage,126)- Ghimj(index,42)*K(index,istage,137))/(Ghimj(index,38));
        K(index,istage,9) = (K(index,istage,9)- Ghimj(index,30)*K(index,istage,42)- Ghimj(index,31)*K(index,istage,43)- Ghimj(index,32)*K(index,istage,52)- Ghimj(index,33)*K(index,istage,54)- Ghimj(index,34)*K(index,istage,55)- Ghimj(index,35)*K(index,istage,75)- Ghimj(index,36)*K(index,istage,120)  - Ghimj(index,37)*K(index,istage,126))/(Ghimj(index,29));
        K(index,istage,8) = (K(index,istage,8)- Ghimj(index,26)*K(index,istage,42)- Ghimj(index,27)*K(index,istage,43)- Ghimj(index,28)*K(index,istage,120))/(Ghimj(index,25));
        K(index,istage,7) = (K(index,istage,7)- Ghimj(index,10)*K(index,istage,41)- Ghimj(index,11)*K(index,istage,42)- Ghimj(index,12)*K(index,istage,43)- Ghimj(index,13)*K(index,istage,44)- Ghimj(index,14)*K(index,istage,45)- Ghimj(index,15)*K(index,istage,52)- Ghimj(index,16)*K(index,istage,53)- Ghimj(index,17)  *K(index,istage,54)- Ghimj(index,18)*K(index,istage,55)- Ghimj(index,19)*K(index,istage,57)- Ghimj(index,20)*K(index,istage,75)- Ghimj(index,21)*K(index,istage,120)- Ghimj(index,22)*K(index,istage,126))/(Ghimj(index,9));
        K(index,istage,6) = K(index,istage,6)/ Ghimj(index,6);
        K(index,istage,5) = K(index,istage,5)/ Ghimj(index,5);
        K(index,istage,4) = K(index,istage,4)/ Ghimj(index,4);
        K(index,istage,3) = K(index,istage,3)/ Ghimj(index,3);
        K(index,istage,2) = K(index,istage,2)/ Ghimj(index,2);
        K(index,istage,1) = K(index,istage,1)/ Ghimj(index,1);
        K(index,istage,0) = K(index,istage,0)/ Ghimj(index,0);
}

__device__ void ros_Solve(double * __restrict__ Ghimj, double * __restrict__ K, int &Nsol, const int istage, const int ros_S)
{

    int index = blockIdx.x*blockDim.x+threadIdx.x;

    #pragma unroll 4 
    for (int i=0;i<LU_NONZERO-16;i+=16){
      prefetch_ll1(&Ghimj(index,i));
    }

    kppSolve(Ghimj, K, istage, ros_S);
    Nsol++;
}

__device__ void kppDecomp(double *Ghimj, int VL_GLO)
{
    double a=0.0;

 double dummy, W_0, W_1, W_2, W_3, W_4, W_5, W_6, W_7, W_8, W_9, W_10, W_11, W_12, W_13, W_14, W_15, W_16, W_17, W_18, W_19, W_20, W_21, W_22, W_23, W_24, W_25, W_26, W_27, W_28, W_29, W_30, W_31, W_32, W_33, W_34, W_35, W_36, W_37, W_38, W_39, W_40, W_41, W_42, W_43, W_44, W_45, W_46, W_47, W_48, W_49, W_50, W_51, W_52, W_53, W_54, W_55, W_56, W_57, W_58, W_59, W_60, W_61, W_62, W_63, W_64, W_65, W_66, W_67, W_68, W_69, W_70, W_71, W_72, W_73, W_74, W_75, W_76, W_77, W_78, W_79, W_80, W_81, W_82, W_83, W_84, W_85, W_86, W_87, W_88, W_89, W_90, W_91, W_92, W_93, W_94, W_95, W_96, W_97, W_98, W_99, W_100, W_101, W_102, W_103, W_104, W_105, W_106, W_107, W_108, W_109, W_110, W_111, W_112, W_113, W_114, W_115, W_116, W_117, W_118, W_119, W_120, W_121, W_122, W_123, W_124, W_125, W_126, W_127, W_128, W_129, W_130, W_131, W_132, W_133, W_134, W_135, W_136, W_137, W_138, W_139, W_140, W_141;

        W_1 = Ghimj(index,7);
        W_2 = Ghimj(index,8);
        W_7 = Ghimj(index,9);
        W_41 = Ghimj(index,10);
        W_42 = Ghimj(index,11);
        W_43 = Ghimj(index,12);
        W_44 = Ghimj(index,13);
        W_45 = Ghimj(index,14);
        W_52 = Ghimj(index,15);
        W_53 = Ghimj(index,16);
        W_54 = Ghimj(index,17);
        W_55 = Ghimj(index,18);
        W_57 = Ghimj(index,19);
        W_75 = Ghimj(index,20);
        W_120 = Ghimj(index,21);
        W_126 = Ghimj(index,22);
        a = - W_1/ Ghimj(index,1);
        W_1 = -a;
        a = - W_2/ Ghimj(index,2);
        W_2 = -a;
        Ghimj(index,7) = W_1;
        Ghimj(index,8) = W_2;
        Ghimj(index,9) = W_7;
        Ghimj(index,10) = W_41;
        Ghimj(index,11) = W_42;
        Ghimj(index,12) = W_43;
        Ghimj(index,13) = W_44;
        Ghimj(index,14) = W_45;
        Ghimj(index,15) = W_52;
        Ghimj(index,16) = W_53;
        Ghimj(index,17) = W_54;
        Ghimj(index,18) = W_55;
        Ghimj(index,19) = W_57;
        Ghimj(index,20) = W_75;
        Ghimj(index,21) = W_120;
        Ghimj(index,22) = W_126;
        W_1 = Ghimj(index,23);
        W_2 = Ghimj(index,24);
        W_8 = Ghimj(index,25);
        W_42 = Ghimj(index,26);
        W_43 = Ghimj(index,27);
        W_120 = Ghimj(index,28);
        a = - W_1/ Ghimj(index,1);
        W_1 = -a;
        a = - W_2/ Ghimj(index,2);
        W_2 = -a;
        Ghimj(index,23) = W_1;
        Ghimj(index,24) = W_2;
        Ghimj(index,25) = W_8;
        Ghimj(index,26) = W_42;
        Ghimj(index,27) = W_43;
        Ghimj(index,28) = W_120;
        W_5 = Ghimj(index,50);
        W_6 = Ghimj(index,51);
        W_14 = Ghimj(index,52);
        W_15 = Ghimj(index,53);
        W_16 = Ghimj(index,54);
        W_17 = Ghimj(index,55);
        W_18 = Ghimj(index,56);
        W_120 = Ghimj(index,57);
        a = - W_5/ Ghimj(index,5);
        W_5 = -a;
        a = - W_6/ Ghimj(index,6);
        W_6 = -a;
        Ghimj(index,50) = W_5;
        Ghimj(index,51) = W_6;
        Ghimj(index,52) = W_14;
        Ghimj(index,53) = W_15;
        Ghimj(index,54) = W_16;
        Ghimj(index,55) = W_17;
        Ghimj(index,56) = W_18;
        Ghimj(index,57) = W_120;
        W_4 = Ghimj(index,67);
        W_19 = Ghimj(index,68);
        a = - W_4/ Ghimj(index,4);
        W_4 = -a;
        Ghimj(index,67) = W_4;
        Ghimj(index,68) = W_19;
        W_1 = Ghimj(index,188);
        W_2 = Ghimj(index,189);
        W_31 = Ghimj(index,190);
        W_53 = Ghimj(index,191);
        W_126 = Ghimj(index,192);
        a = - W_1/ Ghimj(index,1);
        W_1 = -a;
        a = - W_2/ Ghimj(index,2);
        W_2 = -a;
        Ghimj(index,188) = W_1;
        Ghimj(index,189) = W_2;
        Ghimj(index,190) = W_31;
        Ghimj(index,191) = W_53;
        Ghimj(index,192) = W_126;
        W_1 = Ghimj(index,193);
        W_32 = Ghimj(index,194);
        W_41 = Ghimj(index,195);
        W_42 = Ghimj(index,196);
        W_43 = Ghimj(index,197);
        W_57 = Ghimj(index,198);
        W_75 = Ghimj(index,199);
        W_120 = Ghimj(index,200);
        W_126 = Ghimj(index,201);
        a = - W_1/ Ghimj(index,1);
        W_1 = -a;
        Ghimj(index,193) = W_1;
        Ghimj(index,194) = W_32;
        Ghimj(index,195) = W_41;
        Ghimj(index,196) = W_42;
        Ghimj(index,197) = W_43;
        Ghimj(index,198) = W_57;
        Ghimj(index,199) = W_75;
        Ghimj(index,200) = W_120;
        Ghimj(index,201) = W_126;
        W_0 = Ghimj(index,205);
        W_34 = Ghimj(index,206);
        W_50 = Ghimj(index,207);
        W_51 = Ghimj(index,208);
        W_59 = Ghimj(index,209);
        W_60 = Ghimj(index,210);
        W_65 = Ghimj(index,211);
        W_73 = Ghimj(index,212);
        W_76 = Ghimj(index,213);
        W_93 = Ghimj(index,214);
        W_94 = Ghimj(index,215);
        W_99 = Ghimj(index,216);
        W_100 = Ghimj(index,217);
        W_101 = Ghimj(index,218);
        W_102 = Ghimj(index,219);
        W_109 = Ghimj(index,220);
        W_113 = Ghimj(index,221);
        W_114 = Ghimj(index,222);
        W_115 = Ghimj(index,223);
        W_117 = Ghimj(index,224);
        W_121 = Ghimj(index,225);
        W_122 = Ghimj(index,226);
        W_125 = Ghimj(index,227);
        W_126 = Ghimj(index,228);
        W_127 = Ghimj(index,229);
        W_129 = Ghimj(index,230);
        W_133 = Ghimj(index,231);
        W_137 = Ghimj(index,232);
        a = - W_0/ Ghimj(index,0);
        W_0 = -a;
        Ghimj(index,205) = W_0;
        Ghimj(index,206) = W_34;
        Ghimj(index,207) = W_50;
        Ghimj(index,208) = W_51;
        Ghimj(index,209) = W_59;
        Ghimj(index,210) = W_60;
        Ghimj(index,211) = W_65;
        Ghimj(index,212) = W_73;
        Ghimj(index,213) = W_76;
        Ghimj(index,214) = W_93;
        Ghimj(index,215) = W_94;
        Ghimj(index,216) = W_99;
        Ghimj(index,217) = W_100;
        Ghimj(index,218) = W_101;
        Ghimj(index,219) = W_102;
        Ghimj(index,220) = W_109;
        Ghimj(index,221) = W_113;
        Ghimj(index,222) = W_114;
        Ghimj(index,223) = W_115;
        Ghimj(index,224) = W_117;
        Ghimj(index,225) = W_121;
        Ghimj(index,226) = W_122;
        Ghimj(index,227) = W_125;
        Ghimj(index,228) = W_126;
        Ghimj(index,229) = W_127;
        Ghimj(index,230) = W_129;
        Ghimj(index,231) = W_133;
        Ghimj(index,232) = W_137;
        W_59 = Ghimj(index,309);
        W_60 = Ghimj(index,310);
        W_92 = Ghimj(index,311);
        W_120 = Ghimj(index,312);
        W_133 = Ghimj(index,313);
        W_135 = Ghimj(index,314);
        a = - W_59/ Ghimj(index,306);
        W_59 = -a;
        W_133 = W_133+ a *Ghimj(index,307);
        W_135 = W_135+ a *Ghimj(index,308);
        Ghimj(index,309) = W_59;
        Ghimj(index,310) = W_60;
        Ghimj(index,311) = W_92;
        Ghimj(index,312) = W_120;
        Ghimj(index,313) = W_133;
        Ghimj(index,314) = W_135;
        W_61 = Ghimj(index,351);
        W_70 = Ghimj(index,352);
        W_84 = Ghimj(index,353);
        W_87 = Ghimj(index,354);
        W_126 = Ghimj(index,355);
        a = - W_61/ Ghimj(index,315);
        W_61 = -a;
        W_70 = W_70+ a *Ghimj(index,316);
        W_87 = W_87+ a *Ghimj(index,317);
        W_126 = W_126+ a *Ghimj(index,318);
        Ghimj(index,351) = W_61;
        Ghimj(index,352) = W_70;
        Ghimj(index,353) = W_84;
        Ghimj(index,354) = W_87;
        Ghimj(index,355) = W_126;
        W_79 = Ghimj(index,426);
        W_85 = Ghimj(index,427);
        W_102 = Ghimj(index,428);
        W_111 = Ghimj(index,429);
        W_125 = Ghimj(index,430);
        W_126 = Ghimj(index,431);
        W_133 = Ghimj(index,432);
        W_137 = Ghimj(index,433);
        a = - W_79/ Ghimj(index,393);
        W_79 = -a;
        W_102 = W_102+ a *Ghimj(index,394);
        W_126 = W_126+ a *Ghimj(index,395);
        W_137 = W_137+ a *Ghimj(index,396);
        Ghimj(index,426) = W_79;
        Ghimj(index,427) = W_85;
        Ghimj(index,428) = W_102;
        Ghimj(index,429) = W_111;
        Ghimj(index,430) = W_125;
        Ghimj(index,431) = W_126;
        Ghimj(index,432) = W_133;
        Ghimj(index,433) = W_137;
        W_62 = Ghimj(index,434);
        W_69 = Ghimj(index,435);
        W_86 = Ghimj(index,436);
        W_93 = Ghimj(index,437);
        W_125 = Ghimj(index,438);
        W_126 = Ghimj(index,439);
        W_133 = Ghimj(index,440);
        W_137 = Ghimj(index,441);
        a = - W_62/ Ghimj(index,319);
        W_62 = -a;
        W_93 = W_93+ a *Ghimj(index,320);
        W_126 = W_126+ a *Ghimj(index,321);
        W_133 = W_133+ a *Ghimj(index,322);
        a = - W_69/ Ghimj(index,347);
        W_69 = -a;
        W_93 = W_93+ a *Ghimj(index,348);
        W_126 = W_126+ a *Ghimj(index,349);
        W_137 = W_137+ a *Ghimj(index,350);
        Ghimj(index,434) = W_62;
        Ghimj(index,435) = W_69;
        Ghimj(index,436) = W_86;
        Ghimj(index,437) = W_93;
        Ghimj(index,438) = W_125;
        Ghimj(index,439) = W_126;
        Ghimj(index,440) = W_133;
        Ghimj(index,441) = W_137;
        W_70 = Ghimj(index,442);
        W_84 = Ghimj(index,443);
        W_87 = Ghimj(index,444);
        W_92 = Ghimj(index,445);
        W_124 = Ghimj(index,446);
        W_126 = Ghimj(index,447);
        W_135 = Ghimj(index,448);
        W_137 = Ghimj(index,449);
        a = - W_70/ Ghimj(index,352);
        W_70 = -a;
        W_84 = W_84+ a *Ghimj(index,353);
        W_87 = W_87+ a *Ghimj(index,354);
        W_126 = W_126+ a *Ghimj(index,355);
        a = - W_84/ Ghimj(index,421);
        W_84 = -a;
        W_92 = W_92+ a *Ghimj(index,422);
        W_124 = W_124+ a *Ghimj(index,423);
        W_135 = W_135+ a *Ghimj(index,424);
        W_137 = W_137+ a *Ghimj(index,425);
        Ghimj(index,442) = W_70;
        Ghimj(index,443) = W_84;
        Ghimj(index,444) = W_87;
        Ghimj(index,445) = W_92;
        Ghimj(index,446) = W_124;
        Ghimj(index,447) = W_126;
        Ghimj(index,448) = W_135;
        Ghimj(index,449) = W_137;
        W_80 = Ghimj(index,468);
        W_90 = Ghimj(index,469);
        W_100 = Ghimj(index,470);
        W_105 = Ghimj(index,471);
        W_112 = Ghimj(index,472);
        W_116 = Ghimj(index,473);
        W_118 = Ghimj(index,474);
        W_123 = Ghimj(index,475);
        W_127 = Ghimj(index,476);
        W_129 = Ghimj(index,477);
        W_132 = Ghimj(index,478);
        W_134 = Ghimj(index,479);
        W_138 = Ghimj(index,480);
        a = - W_80/ Ghimj(index,397);
        W_80 = -a;
        W_90 = W_90+ a *Ghimj(index,398);
        W_112 = W_112+ a *Ghimj(index,399);
        W_116 = W_116+ a *Ghimj(index,400);
        W_127 = W_127+ a *Ghimj(index,401);
        W_129 = W_129+ a *Ghimj(index,402);
        W_134 = W_134+ a *Ghimj(index,403);
        W_138 = W_138+ a *Ghimj(index,404);
        Ghimj(index,468) = W_80;
        Ghimj(index,469) = W_90;
        Ghimj(index,470) = W_100;
        Ghimj(index,471) = W_105;
        Ghimj(index,472) = W_112;
        Ghimj(index,473) = W_116;
        Ghimj(index,474) = W_118;
        Ghimj(index,475) = W_123;
        Ghimj(index,476) = W_127;
        Ghimj(index,477) = W_129;
        Ghimj(index,478) = W_132;
        Ghimj(index,479) = W_134;
        Ghimj(index,480) = W_138;
        W_47 = Ghimj(index,487);
        W_84 = Ghimj(index,488);
        W_92 = Ghimj(index,489);
        W_124 = Ghimj(index,490);
        W_126 = Ghimj(index,491);
        W_133 = Ghimj(index,492);
        W_135 = Ghimj(index,493);
        W_137 = Ghimj(index,494);
        a = - W_47/ Ghimj(index,276);
        W_47 = -a;
        W_126 = W_126+ a *Ghimj(index,277);
        a = - W_84/ Ghimj(index,421);
        W_84 = -a;
        W_92 = W_92+ a *Ghimj(index,422);
        W_124 = W_124+ a *Ghimj(index,423);
        W_135 = W_135+ a *Ghimj(index,424);
        W_137 = W_137+ a *Ghimj(index,425);
        Ghimj(index,487) = W_47;
        Ghimj(index,488) = W_84;
        Ghimj(index,489) = W_92;
        Ghimj(index,490) = W_124;
        Ghimj(index,491) = W_126;
        Ghimj(index,492) = W_133;
        Ghimj(index,493) = W_135;
        Ghimj(index,494) = W_137;
        W_49 = Ghimj(index,495);
        W_69 = Ghimj(index,496);
        W_93 = Ghimj(index,497);
        W_125 = Ghimj(index,498);
        W_126 = Ghimj(index,499);
        W_133 = Ghimj(index,500);
        W_137 = Ghimj(index,501);
        a = - W_49/ Ghimj(index,280);
        W_49 = -a;
        W_126 = W_126+ a *Ghimj(index,281);
        a = - W_69/ Ghimj(index,347);
        W_69 = -a;
        W_93 = W_93+ a *Ghimj(index,348);
        W_126 = W_126+ a *Ghimj(index,349);
        W_137 = W_137+ a *Ghimj(index,350);
        Ghimj(index,495) = W_49;
        Ghimj(index,496) = W_69;
        Ghimj(index,497) = W_93;
        Ghimj(index,498) = W_125;
        Ghimj(index,499) = W_126;
        Ghimj(index,500) = W_133;
        Ghimj(index,501) = W_137;
        W_72 = Ghimj(index,502);
        W_86 = Ghimj(index,503);
        W_93 = Ghimj(index,504);
        W_94 = Ghimj(index,505);
        W_125 = Ghimj(index,506);
        W_126 = Ghimj(index,507);
        W_133 = Ghimj(index,508);
        W_137 = Ghimj(index,509);
        a = - W_72/ Ghimj(index,360);
        W_72 = -a;
        W_94 = W_94+ a *Ghimj(index,361);
        W_126 = W_126+ a *Ghimj(index,362);
        W_137 = W_137+ a *Ghimj(index,363);
        a = - W_86/ Ghimj(index,436);
        W_86 = -a;
        W_93 = W_93+ a *Ghimj(index,437);
        W_125 = W_125+ a *Ghimj(index,438);
        W_126 = W_126+ a *Ghimj(index,439);
        W_133 = W_133+ a *Ghimj(index,440);
        W_137 = W_137+ a *Ghimj(index,441);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        Ghimj(index,502) = W_72;
        Ghimj(index,503) = W_86;
        Ghimj(index,504) = W_93;
        Ghimj(index,505) = W_94;
        Ghimj(index,506) = W_125;
        Ghimj(index,507) = W_126;
        Ghimj(index,508) = W_133;
        Ghimj(index,509) = W_137;
        W_58 = Ghimj(index,510);
        W_77 = Ghimj(index,511);
        W_82 = Ghimj(index,512);
        W_91 = Ghimj(index,513);
        W_95 = Ghimj(index,514);
        W_96 = Ghimj(index,515);
        W_98 = Ghimj(index,516);
        W_103 = Ghimj(index,517);
        W_106 = Ghimj(index,518);
        W_107 = Ghimj(index,519);
        W_109 = Ghimj(index,520);
        W_110 = Ghimj(index,521);
        W_113 = Ghimj(index,522);
        W_119 = Ghimj(index,523);
        W_121 = Ghimj(index,524);
        W_124 = Ghimj(index,525);
        W_125 = Ghimj(index,526);
        W_126 = Ghimj(index,527);
        W_127 = Ghimj(index,528);
        W_129 = Ghimj(index,529);
        W_130 = Ghimj(index,530);
        W_133 = Ghimj(index,531);
        W_135 = Ghimj(index,532);
        W_136 = Ghimj(index,533);
        W_137 = Ghimj(index,534);
        a = - W_58/ Ghimj(index,303);
        W_58 = -a;
        W_91 = W_91+ a *Ghimj(index,304);
        W_126 = W_126+ a *Ghimj(index,305);
        a = - W_77/ Ghimj(index,382);
        W_77 = -a;
        W_121 = W_121+ a *Ghimj(index,383);
        W_126 = W_126+ a *Ghimj(index,384);
        W_135 = W_135+ a *Ghimj(index,385);
        a = - W_82/ Ghimj(index,412);
        W_82 = -a;
        W_113 = W_113+ a *Ghimj(index,413);
        W_126 = W_126+ a *Ghimj(index,414);
        W_137 = W_137+ a *Ghimj(index,415);
        a = - W_91/ Ghimj(index,481);
        W_91 = -a;
        W_106 = W_106+ a *Ghimj(index,482);
        W_109 = W_109+ a *Ghimj(index,483);
        W_126 = W_126+ a *Ghimj(index,484);
        W_133 = W_133+ a *Ghimj(index,485);
        W_136 = W_136+ a *Ghimj(index,486);
        Ghimj(index,510) = W_58;
        Ghimj(index,511) = W_77;
        Ghimj(index,512) = W_82;
        Ghimj(index,513) = W_91;
        Ghimj(index,514) = W_95;
        Ghimj(index,515) = W_96;
        Ghimj(index,516) = W_98;
        Ghimj(index,517) = W_103;
        Ghimj(index,518) = W_106;
        Ghimj(index,519) = W_107;
        Ghimj(index,520) = W_109;
        Ghimj(index,521) = W_110;
        Ghimj(index,522) = W_113;
        Ghimj(index,523) = W_119;
        Ghimj(index,524) = W_121;
        Ghimj(index,525) = W_124;
        Ghimj(index,526) = W_125;
        Ghimj(index,527) = W_126;
        Ghimj(index,528) = W_127;
        Ghimj(index,529) = W_129;
        Ghimj(index,530) = W_130;
        Ghimj(index,531) = W_133;
        Ghimj(index,532) = W_135;
        Ghimj(index,533) = W_136;
        Ghimj(index,534) = W_137;
        W_72 = Ghimj(index,535);
        W_82 = Ghimj(index,536);
        W_94 = Ghimj(index,537);
        W_96 = Ghimj(index,538);
        W_107 = Ghimj(index,539);
        W_108 = Ghimj(index,540);
        W_109 = Ghimj(index,541);
        W_110 = Ghimj(index,542);
        W_113 = Ghimj(index,543);
        W_124 = Ghimj(index,544);
        W_125 = Ghimj(index,545);
        W_126 = Ghimj(index,546);
        W_133 = Ghimj(index,547);
        W_137 = Ghimj(index,548);
        a = - W_72/ Ghimj(index,360);
        W_72 = -a;
        W_94 = W_94+ a *Ghimj(index,361);
        W_126 = W_126+ a *Ghimj(index,362);
        W_137 = W_137+ a *Ghimj(index,363);
        a = - W_82/ Ghimj(index,412);
        W_82 = -a;
        W_113 = W_113+ a *Ghimj(index,413);
        W_126 = W_126+ a *Ghimj(index,414);
        W_137 = W_137+ a *Ghimj(index,415);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        Ghimj(index,535) = W_72;
        Ghimj(index,536) = W_82;
        Ghimj(index,537) = W_94;
        Ghimj(index,538) = W_96;
        Ghimj(index,539) = W_107;
        Ghimj(index,540) = W_108;
        Ghimj(index,541) = W_109;
        Ghimj(index,542) = W_110;
        Ghimj(index,543) = W_113;
        Ghimj(index,544) = W_124;
        Ghimj(index,545) = W_125;
        Ghimj(index,546) = W_126;
        Ghimj(index,547) = W_133;
        Ghimj(index,548) = W_137;
        W_68 = Ghimj(index,563);
        W_85 = Ghimj(index,564);
        W_99 = Ghimj(index,565);
        W_102 = Ghimj(index,566);
        W_111 = Ghimj(index,567);
        W_125 = Ghimj(index,568);
        W_126 = Ghimj(index,569);
        W_133 = Ghimj(index,570);
        W_137 = Ghimj(index,571);
        a = - W_68/ Ghimj(index,343);
        W_68 = -a;
        W_99 = W_99+ a *Ghimj(index,344);
        W_126 = W_126+ a *Ghimj(index,345);
        W_137 = W_137+ a *Ghimj(index,346);
        a = - W_85/ Ghimj(index,427);
        W_85 = -a;
        W_102 = W_102+ a *Ghimj(index,428);
        W_111 = W_111+ a *Ghimj(index,429);
        W_125 = W_125+ a *Ghimj(index,430);
        W_126 = W_126+ a *Ghimj(index,431);
        W_133 = W_133+ a *Ghimj(index,432);
        W_137 = W_137+ a *Ghimj(index,433);
        Ghimj(index,563) = W_68;
        Ghimj(index,564) = W_85;
        Ghimj(index,565) = W_99;
        Ghimj(index,566) = W_102;
        Ghimj(index,567) = W_111;
        Ghimj(index,568) = W_125;
        Ghimj(index,569) = W_126;
        Ghimj(index,570) = W_133;
        Ghimj(index,571) = W_137;
        W_90 = Ghimj(index,572);
        W_100 = Ghimj(index,573);
        W_105 = Ghimj(index,574);
        W_112 = Ghimj(index,575);
        W_116 = Ghimj(index,576);
        W_118 = Ghimj(index,577);
        W_123 = Ghimj(index,578);
        W_126 = Ghimj(index,579);
        W_127 = Ghimj(index,580);
        W_129 = Ghimj(index,581);
        W_132 = Ghimj(index,582);
        W_134 = Ghimj(index,583);
        W_138 = Ghimj(index,584);
        a = - W_90/ Ghimj(index,469);
        W_90 = -a;
        W_100 = W_100+ a *Ghimj(index,470);
        W_105 = W_105+ a *Ghimj(index,471);
        W_112 = W_112+ a *Ghimj(index,472);
        W_116 = W_116+ a *Ghimj(index,473);
        W_118 = W_118+ a *Ghimj(index,474);
        W_123 = W_123+ a *Ghimj(index,475);
        W_127 = W_127+ a *Ghimj(index,476);
        W_129 = W_129+ a *Ghimj(index,477);
        W_132 = W_132+ a *Ghimj(index,478);
        W_134 = W_134+ a *Ghimj(index,479);
        W_138 = W_138+ a *Ghimj(index,480);
        Ghimj(index,572) = W_90;
        Ghimj(index,573) = W_100;
        Ghimj(index,574) = W_105;
        Ghimj(index,575) = W_112;
        Ghimj(index,576) = W_116;
        Ghimj(index,577) = W_118;
        Ghimj(index,578) = W_123;
        Ghimj(index,579) = W_126;
        Ghimj(index,580) = W_127;
        Ghimj(index,581) = W_129;
        Ghimj(index,582) = W_132;
        Ghimj(index,583) = W_134;
        Ghimj(index,584) = W_138;
        W_83 = Ghimj(index,585);
        W_101 = Ghimj(index,586);
        W_105 = Ghimj(index,587);
        W_114 = Ghimj(index,588);
        W_116 = Ghimj(index,589);
        W_119 = Ghimj(index,590);
        W_123 = Ghimj(index,591);
        W_126 = Ghimj(index,592);
        W_128 = Ghimj(index,593);
        W_130 = Ghimj(index,594);
        W_135 = Ghimj(index,595);
        W_136 = Ghimj(index,596);
        W_138 = Ghimj(index,597);
        a = - W_83/ Ghimj(index,416);
        W_83 = -a;
        W_128 = W_128+ a *Ghimj(index,417);
        W_135 = W_135+ a *Ghimj(index,418);
        W_136 = W_136+ a *Ghimj(index,419);
        W_138 = W_138+ a *Ghimj(index,420);
        Ghimj(index,585) = W_83;
        Ghimj(index,586) = W_101;
        Ghimj(index,587) = W_105;
        Ghimj(index,588) = W_114;
        Ghimj(index,589) = W_116;
        Ghimj(index,590) = W_119;
        Ghimj(index,591) = W_123;
        Ghimj(index,592) = W_126;
        Ghimj(index,593) = W_128;
        Ghimj(index,594) = W_130;
        Ghimj(index,595) = W_135;
        Ghimj(index,596) = W_136;
        Ghimj(index,597) = W_138;
        W_40 = Ghimj(index,598);
        W_79 = Ghimj(index,599);
        W_102 = Ghimj(index,600);
        W_125 = Ghimj(index,601);
        W_126 = Ghimj(index,602);
        W_133 = Ghimj(index,603);
        W_137 = Ghimj(index,604);
        a = - W_40/ Ghimj(index,260);
        W_40 = -a;
        W_126 = W_126+ a *Ghimj(index,261);
        a = - W_79/ Ghimj(index,393);
        W_79 = -a;
        W_102 = W_102+ a *Ghimj(index,394);
        W_126 = W_126+ a *Ghimj(index,395);
        W_137 = W_137+ a *Ghimj(index,396);
        Ghimj(index,598) = W_40;
        Ghimj(index,599) = W_79;
        Ghimj(index,600) = W_102;
        Ghimj(index,601) = W_125;
        Ghimj(index,602) = W_126;
        Ghimj(index,603) = W_133;
        Ghimj(index,604) = W_137;
        W_64 = Ghimj(index,630);
        W_67 = Ghimj(index,631);
        W_82 = Ghimj(index,632);
        W_91 = Ghimj(index,633);
        W_94 = Ghimj(index,634);
        W_106 = Ghimj(index,635);
        W_108 = Ghimj(index,636);
        W_109 = Ghimj(index,637);
        W_113 = Ghimj(index,638);
        W_115 = Ghimj(index,639);
        W_124 = Ghimj(index,640);
        W_125 = Ghimj(index,641);
        W_126 = Ghimj(index,642);
        W_133 = Ghimj(index,643);
        W_135 = Ghimj(index,644);
        W_136 = Ghimj(index,645);
        W_137 = Ghimj(index,646);
        a = - W_64/ Ghimj(index,327);
        W_64 = -a;
        W_113 = W_113+ a *Ghimj(index,328);
        W_126 = W_126+ a *Ghimj(index,329);
        W_135 = W_135+ a *Ghimj(index,330);
        a = - W_67/ Ghimj(index,339);
        W_67 = -a;
        W_115 = W_115+ a *Ghimj(index,340);
        W_126 = W_126+ a *Ghimj(index,341);
        W_137 = W_137+ a *Ghimj(index,342);
        a = - W_82/ Ghimj(index,412);
        W_82 = -a;
        W_113 = W_113+ a *Ghimj(index,413);
        W_126 = W_126+ a *Ghimj(index,414);
        W_137 = W_137+ a *Ghimj(index,415);
        a = - W_91/ Ghimj(index,481);
        W_91 = -a;
        W_106 = W_106+ a *Ghimj(index,482);
        W_109 = W_109+ a *Ghimj(index,483);
        W_126 = W_126+ a *Ghimj(index,484);
        W_133 = W_133+ a *Ghimj(index,485);
        W_136 = W_136+ a *Ghimj(index,486);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        Ghimj(index,630) = W_64;
        Ghimj(index,631) = W_67;
        Ghimj(index,632) = W_82;
        Ghimj(index,633) = W_91;
        Ghimj(index,634) = W_94;
        Ghimj(index,635) = W_106;
        Ghimj(index,636) = W_108;
        Ghimj(index,637) = W_109;
        Ghimj(index,638) = W_113;
        Ghimj(index,639) = W_115;
        Ghimj(index,640) = W_124;
        Ghimj(index,641) = W_125;
        Ghimj(index,642) = W_126;
        Ghimj(index,643) = W_133;
        Ghimj(index,644) = W_135;
        Ghimj(index,645) = W_136;
        Ghimj(index,646) = W_137;
        W_106 = Ghimj(index,647);
        W_109 = Ghimj(index,648);
        W_124 = Ghimj(index,649);
        W_125 = Ghimj(index,650);
        W_126 = Ghimj(index,651);
        W_133 = Ghimj(index,652);
        W_136 = Ghimj(index,653);
        W_137 = Ghimj(index,654);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        Ghimj(index,647) = W_106;
        Ghimj(index,648) = W_109;
        Ghimj(index,649) = W_124;
        Ghimj(index,650) = W_125;
        Ghimj(index,651) = W_126;
        Ghimj(index,652) = W_133;
        Ghimj(index,653) = W_136;
        Ghimj(index,654) = W_137;
        W_66 = Ghimj(index,655);
        W_91 = Ghimj(index,656);
        W_106 = Ghimj(index,657);
        W_109 = Ghimj(index,658);
        W_110 = Ghimj(index,659);
        W_124 = Ghimj(index,660);
        W_125 = Ghimj(index,661);
        W_126 = Ghimj(index,662);
        W_133 = Ghimj(index,663);
        W_136 = Ghimj(index,664);
        W_137 = Ghimj(index,665);
        a = - W_66/ Ghimj(index,335);
        W_66 = -a;
        W_109 = W_109+ a *Ghimj(index,336);
        W_126 = W_126+ a *Ghimj(index,337);
        W_137 = W_137+ a *Ghimj(index,338);
        a = - W_91/ Ghimj(index,481);
        W_91 = -a;
        W_106 = W_106+ a *Ghimj(index,482);
        W_109 = W_109+ a *Ghimj(index,483);
        W_126 = W_126+ a *Ghimj(index,484);
        W_133 = W_133+ a *Ghimj(index,485);
        W_136 = W_136+ a *Ghimj(index,486);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        Ghimj(index,655) = W_66;
        Ghimj(index,656) = W_91;
        Ghimj(index,657) = W_106;
        Ghimj(index,658) = W_109;
        Ghimj(index,659) = W_110;
        Ghimj(index,660) = W_124;
        Ghimj(index,661) = W_125;
        Ghimj(index,662) = W_126;
        Ghimj(index,663) = W_133;
        Ghimj(index,664) = W_136;
        Ghimj(index,665) = W_137;
        W_99 = Ghimj(index,666);
        W_102 = Ghimj(index,667);
        W_107 = Ghimj(index,668);
        W_111 = Ghimj(index,669);
        W_115 = Ghimj(index,670);
        W_124 = Ghimj(index,671);
        W_125 = Ghimj(index,672);
        W_126 = Ghimj(index,673);
        W_133 = Ghimj(index,674);
        W_136 = Ghimj(index,675);
        W_137 = Ghimj(index,676);
        a = - W_99/ Ghimj(index,565);
        W_99 = -a;
        W_102 = W_102+ a *Ghimj(index,566);
        W_111 = W_111+ a *Ghimj(index,567);
        W_125 = W_125+ a *Ghimj(index,568);
        W_126 = W_126+ a *Ghimj(index,569);
        W_133 = W_133+ a *Ghimj(index,570);
        W_137 = W_137+ a *Ghimj(index,571);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        Ghimj(index,666) = W_99;
        Ghimj(index,667) = W_102;
        Ghimj(index,668) = W_107;
        Ghimj(index,669) = W_111;
        Ghimj(index,670) = W_115;
        Ghimj(index,671) = W_124;
        Ghimj(index,672) = W_125;
        Ghimj(index,673) = W_126;
        Ghimj(index,674) = W_133;
        Ghimj(index,675) = W_136;
        Ghimj(index,676) = W_137;
        W_64 = Ghimj(index,685);
        W_82 = Ghimj(index,686);
        W_106 = Ghimj(index,687);
        W_110 = Ghimj(index,688);
        W_113 = Ghimj(index,689);
        W_124 = Ghimj(index,690);
        W_125 = Ghimj(index,691);
        W_126 = Ghimj(index,692);
        W_133 = Ghimj(index,693);
        W_135 = Ghimj(index,694);
        W_136 = Ghimj(index,695);
        W_137 = Ghimj(index,696);
        a = - W_64/ Ghimj(index,327);
        W_64 = -a;
        W_113 = W_113+ a *Ghimj(index,328);
        W_126 = W_126+ a *Ghimj(index,329);
        W_135 = W_135+ a *Ghimj(index,330);
        a = - W_82/ Ghimj(index,412);
        W_82 = -a;
        W_113 = W_113+ a *Ghimj(index,413);
        W_126 = W_126+ a *Ghimj(index,414);
        W_137 = W_137+ a *Ghimj(index,415);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_110/ Ghimj(index,659);
        W_110 = -a;
        W_124 = W_124+ a *Ghimj(index,660);
        W_125 = W_125+ a *Ghimj(index,661);
        W_126 = W_126+ a *Ghimj(index,662);
        W_133 = W_133+ a *Ghimj(index,663);
        W_136 = W_136+ a *Ghimj(index,664);
        W_137 = W_137+ a *Ghimj(index,665);
        Ghimj(index,685) = W_64;
        Ghimj(index,686) = W_82;
        Ghimj(index,687) = W_106;
        Ghimj(index,688) = W_110;
        Ghimj(index,689) = W_113;
        Ghimj(index,690) = W_124;
        Ghimj(index,691) = W_125;
        Ghimj(index,692) = W_126;
        Ghimj(index,693) = W_133;
        Ghimj(index,694) = W_135;
        Ghimj(index,695) = W_136;
        Ghimj(index,696) = W_137;
        W_67 = Ghimj(index,703);
        W_103 = Ghimj(index,704);
        W_107 = Ghimj(index,705);
        W_115 = Ghimj(index,706);
        W_124 = Ghimj(index,707);
        W_126 = Ghimj(index,708);
        W_127 = Ghimj(index,709);
        W_129 = Ghimj(index,710);
        W_133 = Ghimj(index,711);
        W_136 = Ghimj(index,712);
        W_137 = Ghimj(index,713);
        a = - W_67/ Ghimj(index,339);
        W_67 = -a;
        W_115 = W_115+ a *Ghimj(index,340);
        W_126 = W_126+ a *Ghimj(index,341);
        W_137 = W_137+ a *Ghimj(index,342);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        Ghimj(index,703) = W_67;
        Ghimj(index,704) = W_103;
        Ghimj(index,705) = W_107;
        Ghimj(index,706) = W_115;
        Ghimj(index,707) = W_124;
        Ghimj(index,708) = W_126;
        Ghimj(index,709) = W_127;
        Ghimj(index,710) = W_129;
        Ghimj(index,711) = W_133;
        Ghimj(index,712) = W_136;
        Ghimj(index,713) = W_137;
        W_48 = Ghimj(index,722);
        W_49 = Ghimj(index,723);
        W_71 = Ghimj(index,724);
        W_79 = Ghimj(index,725);
        W_85 = Ghimj(index,726);
        W_102 = Ghimj(index,727);
        W_107 = Ghimj(index,728);
        W_111 = Ghimj(index,729);
        W_115 = Ghimj(index,730);
        W_117 = Ghimj(index,731);
        W_121 = Ghimj(index,732);
        W_124 = Ghimj(index,733);
        W_125 = Ghimj(index,734);
        W_126 = Ghimj(index,735);
        W_127 = Ghimj(index,736);
        W_129 = Ghimj(index,737);
        W_133 = Ghimj(index,738);
        W_136 = Ghimj(index,739);
        W_137 = Ghimj(index,740);
        a = - W_48/ Ghimj(index,278);
        W_48 = -a;
        W_126 = W_126+ a *Ghimj(index,279);
        a = - W_49/ Ghimj(index,280);
        W_49 = -a;
        W_126 = W_126+ a *Ghimj(index,281);
        a = - W_71/ Ghimj(index,356);
        W_71 = -a;
        W_117 = W_117+ a *Ghimj(index,357);
        W_126 = W_126+ a *Ghimj(index,358);
        W_137 = W_137+ a *Ghimj(index,359);
        a = - W_79/ Ghimj(index,393);
        W_79 = -a;
        W_102 = W_102+ a *Ghimj(index,394);
        W_126 = W_126+ a *Ghimj(index,395);
        W_137 = W_137+ a *Ghimj(index,396);
        a = - W_85/ Ghimj(index,427);
        W_85 = -a;
        W_102 = W_102+ a *Ghimj(index,428);
        W_111 = W_111+ a *Ghimj(index,429);
        W_125 = W_125+ a *Ghimj(index,430);
        W_126 = W_126+ a *Ghimj(index,431);
        W_133 = W_133+ a *Ghimj(index,432);
        W_137 = W_137+ a *Ghimj(index,433);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_111/ Ghimj(index,669);
        W_111 = -a;
        W_115 = W_115+ a *Ghimj(index,670);
        W_124 = W_124+ a *Ghimj(index,671);
        W_125 = W_125+ a *Ghimj(index,672);
        W_126 = W_126+ a *Ghimj(index,673);
        W_133 = W_133+ a *Ghimj(index,674);
        W_136 = W_136+ a *Ghimj(index,675);
        W_137 = W_137+ a *Ghimj(index,676);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        Ghimj(index,722) = W_48;
        Ghimj(index,723) = W_49;
        Ghimj(index,724) = W_71;
        Ghimj(index,725) = W_79;
        Ghimj(index,726) = W_85;
        Ghimj(index,727) = W_102;
        Ghimj(index,728) = W_107;
        Ghimj(index,729) = W_111;
        Ghimj(index,730) = W_115;
        Ghimj(index,731) = W_117;
        Ghimj(index,732) = W_121;
        Ghimj(index,733) = W_124;
        Ghimj(index,734) = W_125;
        Ghimj(index,735) = W_126;
        Ghimj(index,736) = W_127;
        Ghimj(index,737) = W_129;
        Ghimj(index,738) = W_133;
        Ghimj(index,739) = W_136;
        Ghimj(index,740) = W_137;
        W_100 = Ghimj(index,741);
        W_105 = Ghimj(index,742);
        W_112 = Ghimj(index,743);
        W_116 = Ghimj(index,744);
        W_118 = Ghimj(index,745);
        W_123 = Ghimj(index,746);
        W_125 = Ghimj(index,747);
        W_126 = Ghimj(index,748);
        W_127 = Ghimj(index,749);
        W_128 = Ghimj(index,750);
        W_129 = Ghimj(index,751);
        W_131 = Ghimj(index,752);
        W_132 = Ghimj(index,753);
        W_134 = Ghimj(index,754);
        W_135 = Ghimj(index,755);
        W_137 = Ghimj(index,756);
        W_138 = Ghimj(index,757);
        a = - W_100/ Ghimj(index,573);
        W_100 = -a;
        W_105 = W_105+ a *Ghimj(index,574);
        W_112 = W_112+ a *Ghimj(index,575);
        W_116 = W_116+ a *Ghimj(index,576);
        W_118 = W_118+ a *Ghimj(index,577);
        W_123 = W_123+ a *Ghimj(index,578);
        W_126 = W_126+ a *Ghimj(index,579);
        W_127 = W_127+ a *Ghimj(index,580);
        W_129 = W_129+ a *Ghimj(index,581);
        W_132 = W_132+ a *Ghimj(index,582);
        W_134 = W_134+ a *Ghimj(index,583);
        W_138 = W_138+ a *Ghimj(index,584);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_112/ Ghimj(index,677);
        W_112 = -a;
        W_116 = W_116+ a *Ghimj(index,678);
        W_123 = W_123+ a *Ghimj(index,679);
        W_126 = W_126+ a *Ghimj(index,680);
        W_128 = W_128+ a *Ghimj(index,681);
        W_134 = W_134+ a *Ghimj(index,682);
        W_137 = W_137+ a *Ghimj(index,683);
        W_138 = W_138+ a *Ghimj(index,684);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        Ghimj(index,741) = W_100;
        Ghimj(index,742) = W_105;
        Ghimj(index,743) = W_112;
        Ghimj(index,744) = W_116;
        Ghimj(index,745) = W_118;
        Ghimj(index,746) = W_123;
        Ghimj(index,747) = W_125;
        Ghimj(index,748) = W_126;
        Ghimj(index,749) = W_127;
        Ghimj(index,750) = W_128;
        Ghimj(index,751) = W_129;
        Ghimj(index,752) = W_131;
        Ghimj(index,753) = W_132;
        Ghimj(index,754) = W_134;
        Ghimj(index,755) = W_135;
        Ghimj(index,756) = W_137;
        Ghimj(index,757) = W_138;
        W_68 = Ghimj(index,758);
        W_71 = Ghimj(index,759);
        W_79 = Ghimj(index,760);
        W_99 = Ghimj(index,761);
        W_102 = Ghimj(index,762);
        W_107 = Ghimj(index,763);
        W_111 = Ghimj(index,764);
        W_115 = Ghimj(index,765);
        W_117 = Ghimj(index,766);
        W_119 = Ghimj(index,767);
        W_121 = Ghimj(index,768);
        W_124 = Ghimj(index,769);
        W_125 = Ghimj(index,770);
        W_126 = Ghimj(index,771);
        W_127 = Ghimj(index,772);
        W_129 = Ghimj(index,773);
        W_133 = Ghimj(index,774);
        W_136 = Ghimj(index,775);
        W_137 = Ghimj(index,776);
        a = - W_68/ Ghimj(index,343);
        W_68 = -a;
        W_99 = W_99+ a *Ghimj(index,344);
        W_126 = W_126+ a *Ghimj(index,345);
        W_137 = W_137+ a *Ghimj(index,346);
        a = - W_71/ Ghimj(index,356);
        W_71 = -a;
        W_117 = W_117+ a *Ghimj(index,357);
        W_126 = W_126+ a *Ghimj(index,358);
        W_137 = W_137+ a *Ghimj(index,359);
        a = - W_79/ Ghimj(index,393);
        W_79 = -a;
        W_102 = W_102+ a *Ghimj(index,394);
        W_126 = W_126+ a *Ghimj(index,395);
        W_137 = W_137+ a *Ghimj(index,396);
        a = - W_99/ Ghimj(index,565);
        W_99 = -a;
        W_102 = W_102+ a *Ghimj(index,566);
        W_111 = W_111+ a *Ghimj(index,567);
        W_125 = W_125+ a *Ghimj(index,568);
        W_126 = W_126+ a *Ghimj(index,569);
        W_133 = W_133+ a *Ghimj(index,570);
        W_137 = W_137+ a *Ghimj(index,571);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_111/ Ghimj(index,669);
        W_111 = -a;
        W_115 = W_115+ a *Ghimj(index,670);
        W_124 = W_124+ a *Ghimj(index,671);
        W_125 = W_125+ a *Ghimj(index,672);
        W_126 = W_126+ a *Ghimj(index,673);
        W_133 = W_133+ a *Ghimj(index,674);
        W_136 = W_136+ a *Ghimj(index,675);
        W_137 = W_137+ a *Ghimj(index,676);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        Ghimj(index,758) = W_68;
        Ghimj(index,759) = W_71;
        Ghimj(index,760) = W_79;
        Ghimj(index,761) = W_99;
        Ghimj(index,762) = W_102;
        Ghimj(index,763) = W_107;
        Ghimj(index,764) = W_111;
        Ghimj(index,765) = W_115;
        Ghimj(index,766) = W_117;
        Ghimj(index,767) = W_119;
        Ghimj(index,768) = W_121;
        Ghimj(index,769) = W_124;
        Ghimj(index,770) = W_125;
        Ghimj(index,771) = W_126;
        Ghimj(index,772) = W_127;
        Ghimj(index,773) = W_129;
        Ghimj(index,774) = W_133;
        Ghimj(index,775) = W_136;
        Ghimj(index,776) = W_137;
        W_41 = Ghimj(index,777);
        W_42 = Ghimj(index,778);
        W_43 = Ghimj(index,779);
        W_57 = Ghimj(index,780);
        W_60 = Ghimj(index,781);
        W_75 = Ghimj(index,782);
        W_92 = Ghimj(index,783);
        W_97 = Ghimj(index,784);
        W_98 = Ghimj(index,785);
        W_107 = Ghimj(index,786);
        W_120 = Ghimj(index,787);
        W_122 = Ghimj(index,788);
        W_124 = Ghimj(index,789);
        W_126 = Ghimj(index,790);
        W_127 = Ghimj(index,791);
        W_128 = Ghimj(index,792);
        W_130 = Ghimj(index,793);
        W_133 = Ghimj(index,794);
        W_135 = Ghimj(index,795);
        W_136 = Ghimj(index,796);
        W_137 = Ghimj(index,797);
        a = - W_41/ Ghimj(index,262);
        W_41 = -a;
        W_120 = W_120+ a *Ghimj(index,263);
        a = - W_42/ Ghimj(index,264);
        W_42 = -a;
        W_120 = W_120+ a *Ghimj(index,265);
        a = - W_43/ Ghimj(index,266);
        W_43 = -a;
        W_120 = W_120+ a *Ghimj(index,267);
        a = - W_57/ Ghimj(index,300);
        W_57 = -a;
        W_120 = W_120+ a *Ghimj(index,301);
        W_126 = W_126+ a *Ghimj(index,302);
        a = - W_60/ Ghimj(index,310);
        W_60 = -a;
        W_92 = W_92+ a *Ghimj(index,311);
        W_120 = W_120+ a *Ghimj(index,312);
        W_133 = W_133+ a *Ghimj(index,313);
        W_135 = W_135+ a *Ghimj(index,314);
        a = - W_75/ Ghimj(index,374);
        W_75 = -a;
        W_120 = W_120+ a *Ghimj(index,375);
        W_126 = W_126+ a *Ghimj(index,376);
        a = - W_92/ Ghimj(index,489);
        W_92 = -a;
        W_124 = W_124+ a *Ghimj(index,490);
        W_126 = W_126+ a *Ghimj(index,491);
        W_133 = W_133+ a *Ghimj(index,492);
        W_135 = W_135+ a *Ghimj(index,493);
        W_137 = W_137+ a *Ghimj(index,494);
        a = - W_97/ Ghimj(index,549);
        W_97 = -a;
        W_98 = W_98+ a *Ghimj(index,550);
        W_120 = W_120+ a *Ghimj(index,551);
        W_122 = W_122+ a *Ghimj(index,552);
        W_126 = W_126+ a *Ghimj(index,553);
        W_127 = W_127+ a *Ghimj(index,554);
        W_130 = W_130+ a *Ghimj(index,555);
        W_137 = W_137+ a *Ghimj(index,556);
        a = - W_98/ Ghimj(index,557);
        W_98 = -a;
        W_107 = W_107+ a *Ghimj(index,558);
        W_120 = W_120+ a *Ghimj(index,559);
        W_124 = W_124+ a *Ghimj(index,560);
        W_126 = W_126+ a *Ghimj(index,561);
        W_127 = W_127+ a *Ghimj(index,562);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        Ghimj(index,777) = W_41;
        Ghimj(index,778) = W_42;
        Ghimj(index,779) = W_43;
        Ghimj(index,780) = W_57;
        Ghimj(index,781) = W_60;
        Ghimj(index,782) = W_75;
        Ghimj(index,783) = W_92;
        Ghimj(index,784) = W_97;
        Ghimj(index,785) = W_98;
        Ghimj(index,786) = W_107;
        Ghimj(index,787) = W_120;
        Ghimj(index,788) = W_122;
        Ghimj(index,789) = W_124;
        Ghimj(index,790) = W_126;
        Ghimj(index,791) = W_127;
        Ghimj(index,792) = W_128;
        Ghimj(index,793) = W_130;
        Ghimj(index,794) = W_133;
        Ghimj(index,795) = W_135;
        Ghimj(index,796) = W_136;
        Ghimj(index,797) = W_137;
        W_38 = Ghimj(index,798);
        W_63 = Ghimj(index,799);
        W_68 = Ghimj(index,800);
        W_72 = Ghimj(index,801);
        W_77 = Ghimj(index,802);
        W_82 = Ghimj(index,803);
        W_85 = Ghimj(index,804);
        W_86 = Ghimj(index,805);
        W_93 = Ghimj(index,806);
        W_94 = Ghimj(index,807);
        W_96 = Ghimj(index,808);
        W_99 = Ghimj(index,809);
        W_102 = Ghimj(index,810);
        W_106 = Ghimj(index,811);
        W_107 = Ghimj(index,812);
        W_108 = Ghimj(index,813);
        W_109 = Ghimj(index,814);
        W_110 = Ghimj(index,815);
        W_111 = Ghimj(index,816);
        W_113 = Ghimj(index,817);
        W_115 = Ghimj(index,818);
        W_117 = Ghimj(index,819);
        W_119 = Ghimj(index,820);
        W_121 = Ghimj(index,821);
        W_124 = Ghimj(index,822);
        W_125 = Ghimj(index,823);
        W_126 = Ghimj(index,824);
        W_127 = Ghimj(index,825);
        W_129 = Ghimj(index,826);
        W_133 = Ghimj(index,827);
        W_135 = Ghimj(index,828);
        W_136 = Ghimj(index,829);
        W_137 = Ghimj(index,830);
        a = - W_38/ Ghimj(index,255);
        W_38 = -a;
        W_68 = W_68+ a *Ghimj(index,256);
        W_126 = W_126+ a *Ghimj(index,257);
        a = - W_63/ Ghimj(index,323);
        W_63 = -a;
        W_121 = W_121+ a *Ghimj(index,324);
        W_126 = W_126+ a *Ghimj(index,325);
        W_137 = W_137+ a *Ghimj(index,326);
        a = - W_68/ Ghimj(index,343);
        W_68 = -a;
        W_99 = W_99+ a *Ghimj(index,344);
        W_126 = W_126+ a *Ghimj(index,345);
        W_137 = W_137+ a *Ghimj(index,346);
        a = - W_72/ Ghimj(index,360);
        W_72 = -a;
        W_94 = W_94+ a *Ghimj(index,361);
        W_126 = W_126+ a *Ghimj(index,362);
        W_137 = W_137+ a *Ghimj(index,363);
        a = - W_77/ Ghimj(index,382);
        W_77 = -a;
        W_121 = W_121+ a *Ghimj(index,383);
        W_126 = W_126+ a *Ghimj(index,384);
        W_135 = W_135+ a *Ghimj(index,385);
        a = - W_82/ Ghimj(index,412);
        W_82 = -a;
        W_113 = W_113+ a *Ghimj(index,413);
        W_126 = W_126+ a *Ghimj(index,414);
        W_137 = W_137+ a *Ghimj(index,415);
        a = - W_85/ Ghimj(index,427);
        W_85 = -a;
        W_102 = W_102+ a *Ghimj(index,428);
        W_111 = W_111+ a *Ghimj(index,429);
        W_125 = W_125+ a *Ghimj(index,430);
        W_126 = W_126+ a *Ghimj(index,431);
        W_133 = W_133+ a *Ghimj(index,432);
        W_137 = W_137+ a *Ghimj(index,433);
        a = - W_86/ Ghimj(index,436);
        W_86 = -a;
        W_93 = W_93+ a *Ghimj(index,437);
        W_125 = W_125+ a *Ghimj(index,438);
        W_126 = W_126+ a *Ghimj(index,439);
        W_133 = W_133+ a *Ghimj(index,440);
        W_137 = W_137+ a *Ghimj(index,441);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        a = - W_96/ Ghimj(index,538);
        W_96 = -a;
        W_107 = W_107+ a *Ghimj(index,539);
        W_108 = W_108+ a *Ghimj(index,540);
        W_109 = W_109+ a *Ghimj(index,541);
        W_110 = W_110+ a *Ghimj(index,542);
        W_113 = W_113+ a *Ghimj(index,543);
        W_124 = W_124+ a *Ghimj(index,544);
        W_125 = W_125+ a *Ghimj(index,545);
        W_126 = W_126+ a *Ghimj(index,546);
        W_133 = W_133+ a *Ghimj(index,547);
        W_137 = W_137+ a *Ghimj(index,548);
        a = - W_99/ Ghimj(index,565);
        W_99 = -a;
        W_102 = W_102+ a *Ghimj(index,566);
        W_111 = W_111+ a *Ghimj(index,567);
        W_125 = W_125+ a *Ghimj(index,568);
        W_126 = W_126+ a *Ghimj(index,569);
        W_133 = W_133+ a *Ghimj(index,570);
        W_137 = W_137+ a *Ghimj(index,571);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_108/ Ghimj(index,636);
        W_108 = -a;
        W_109 = W_109+ a *Ghimj(index,637);
        W_113 = W_113+ a *Ghimj(index,638);
        W_115 = W_115+ a *Ghimj(index,639);
        W_124 = W_124+ a *Ghimj(index,640);
        W_125 = W_125+ a *Ghimj(index,641);
        W_126 = W_126+ a *Ghimj(index,642);
        W_133 = W_133+ a *Ghimj(index,643);
        W_135 = W_135+ a *Ghimj(index,644);
        W_136 = W_136+ a *Ghimj(index,645);
        W_137 = W_137+ a *Ghimj(index,646);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        a = - W_110/ Ghimj(index,659);
        W_110 = -a;
        W_124 = W_124+ a *Ghimj(index,660);
        W_125 = W_125+ a *Ghimj(index,661);
        W_126 = W_126+ a *Ghimj(index,662);
        W_133 = W_133+ a *Ghimj(index,663);
        W_136 = W_136+ a *Ghimj(index,664);
        W_137 = W_137+ a *Ghimj(index,665);
        a = - W_111/ Ghimj(index,669);
        W_111 = -a;
        W_115 = W_115+ a *Ghimj(index,670);
        W_124 = W_124+ a *Ghimj(index,671);
        W_125 = W_125+ a *Ghimj(index,672);
        W_126 = W_126+ a *Ghimj(index,673);
        W_133 = W_133+ a *Ghimj(index,674);
        W_136 = W_136+ a *Ghimj(index,675);
        W_137 = W_137+ a *Ghimj(index,676);
        a = - W_113/ Ghimj(index,689);
        W_113 = -a;
        W_124 = W_124+ a *Ghimj(index,690);
        W_125 = W_125+ a *Ghimj(index,691);
        W_126 = W_126+ a *Ghimj(index,692);
        W_133 = W_133+ a *Ghimj(index,693);
        W_135 = W_135+ a *Ghimj(index,694);
        W_136 = W_136+ a *Ghimj(index,695);
        W_137 = W_137+ a *Ghimj(index,696);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        Ghimj(index,798) = W_38;
        Ghimj(index,799) = W_63;
        Ghimj(index,800) = W_68;
        Ghimj(index,801) = W_72;
        Ghimj(index,802) = W_77;
        Ghimj(index,803) = W_82;
        Ghimj(index,804) = W_85;
        Ghimj(index,805) = W_86;
        Ghimj(index,806) = W_93;
        Ghimj(index,807) = W_94;
        Ghimj(index,808) = W_96;
        Ghimj(index,809) = W_99;
        Ghimj(index,810) = W_102;
        Ghimj(index,811) = W_106;
        Ghimj(index,812) = W_107;
        Ghimj(index,813) = W_108;
        Ghimj(index,814) = W_109;
        Ghimj(index,815) = W_110;
        Ghimj(index,816) = W_111;
        Ghimj(index,817) = W_113;
        Ghimj(index,818) = W_115;
        Ghimj(index,819) = W_117;
        Ghimj(index,820) = W_119;
        Ghimj(index,821) = W_121;
        Ghimj(index,822) = W_124;
        Ghimj(index,823) = W_125;
        Ghimj(index,824) = W_126;
        Ghimj(index,825) = W_127;
        Ghimj(index,826) = W_129;
        Ghimj(index,827) = W_133;
        Ghimj(index,828) = W_135;
        Ghimj(index,829) = W_136;
        Ghimj(index,830) = W_137;
        W_75 = Ghimj(index,831);
        W_95 = Ghimj(index,832);
        W_96 = Ghimj(index,833);
        W_97 = Ghimj(index,834);
        W_98 = Ghimj(index,835);
        W_103 = Ghimj(index,836);
        W_106 = Ghimj(index,837);
        W_107 = Ghimj(index,838);
        W_108 = Ghimj(index,839);
        W_109 = Ghimj(index,840);
        W_110 = Ghimj(index,841);
        W_113 = Ghimj(index,842);
        W_115 = Ghimj(index,843);
        W_119 = Ghimj(index,844);
        W_120 = Ghimj(index,845);
        W_121 = Ghimj(index,846);
        W_122 = Ghimj(index,847);
        W_124 = Ghimj(index,848);
        W_125 = Ghimj(index,849);
        W_126 = Ghimj(index,850);
        W_127 = Ghimj(index,851);
        W_128 = Ghimj(index,852);
        W_129 = Ghimj(index,853);
        W_130 = Ghimj(index,854);
        W_131 = Ghimj(index,855);
        W_133 = Ghimj(index,856);
        W_135 = Ghimj(index,857);
        W_136 = Ghimj(index,858);
        W_137 = Ghimj(index,859);
        W_138 = Ghimj(index,860);
        a = - W_75/ Ghimj(index,374);
        W_75 = -a;
        W_120 = W_120+ a *Ghimj(index,375);
        W_126 = W_126+ a *Ghimj(index,376);
        a = - W_95/ Ghimj(index,514);
        W_95 = -a;
        W_96 = W_96+ a *Ghimj(index,515);
        W_98 = W_98+ a *Ghimj(index,516);
        W_103 = W_103+ a *Ghimj(index,517);
        W_106 = W_106+ a *Ghimj(index,518);
        W_107 = W_107+ a *Ghimj(index,519);
        W_109 = W_109+ a *Ghimj(index,520);
        W_110 = W_110+ a *Ghimj(index,521);
        W_113 = W_113+ a *Ghimj(index,522);
        W_119 = W_119+ a *Ghimj(index,523);
        W_121 = W_121+ a *Ghimj(index,524);
        W_124 = W_124+ a *Ghimj(index,525);
        W_125 = W_125+ a *Ghimj(index,526);
        W_126 = W_126+ a *Ghimj(index,527);
        W_127 = W_127+ a *Ghimj(index,528);
        W_129 = W_129+ a *Ghimj(index,529);
        W_130 = W_130+ a *Ghimj(index,530);
        W_133 = W_133+ a *Ghimj(index,531);
        W_135 = W_135+ a *Ghimj(index,532);
        W_136 = W_136+ a *Ghimj(index,533);
        W_137 = W_137+ a *Ghimj(index,534);
        a = - W_96/ Ghimj(index,538);
        W_96 = -a;
        W_107 = W_107+ a *Ghimj(index,539);
        W_108 = W_108+ a *Ghimj(index,540);
        W_109 = W_109+ a *Ghimj(index,541);
        W_110 = W_110+ a *Ghimj(index,542);
        W_113 = W_113+ a *Ghimj(index,543);
        W_124 = W_124+ a *Ghimj(index,544);
        W_125 = W_125+ a *Ghimj(index,545);
        W_126 = W_126+ a *Ghimj(index,546);
        W_133 = W_133+ a *Ghimj(index,547);
        W_137 = W_137+ a *Ghimj(index,548);
        a = - W_97/ Ghimj(index,549);
        W_97 = -a;
        W_98 = W_98+ a *Ghimj(index,550);
        W_120 = W_120+ a *Ghimj(index,551);
        W_122 = W_122+ a *Ghimj(index,552);
        W_126 = W_126+ a *Ghimj(index,553);
        W_127 = W_127+ a *Ghimj(index,554);
        W_130 = W_130+ a *Ghimj(index,555);
        W_137 = W_137+ a *Ghimj(index,556);
        a = - W_98/ Ghimj(index,557);
        W_98 = -a;
        W_107 = W_107+ a *Ghimj(index,558);
        W_120 = W_120+ a *Ghimj(index,559);
        W_124 = W_124+ a *Ghimj(index,560);
        W_126 = W_126+ a *Ghimj(index,561);
        W_127 = W_127+ a *Ghimj(index,562);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_108/ Ghimj(index,636);
        W_108 = -a;
        W_109 = W_109+ a *Ghimj(index,637);
        W_113 = W_113+ a *Ghimj(index,638);
        W_115 = W_115+ a *Ghimj(index,639);
        W_124 = W_124+ a *Ghimj(index,640);
        W_125 = W_125+ a *Ghimj(index,641);
        W_126 = W_126+ a *Ghimj(index,642);
        W_133 = W_133+ a *Ghimj(index,643);
        W_135 = W_135+ a *Ghimj(index,644);
        W_136 = W_136+ a *Ghimj(index,645);
        W_137 = W_137+ a *Ghimj(index,646);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        a = - W_110/ Ghimj(index,659);
        W_110 = -a;
        W_124 = W_124+ a *Ghimj(index,660);
        W_125 = W_125+ a *Ghimj(index,661);
        W_126 = W_126+ a *Ghimj(index,662);
        W_133 = W_133+ a *Ghimj(index,663);
        W_136 = W_136+ a *Ghimj(index,664);
        W_137 = W_137+ a *Ghimj(index,665);
        a = - W_113/ Ghimj(index,689);
        W_113 = -a;
        W_124 = W_124+ a *Ghimj(index,690);
        W_125 = W_125+ a *Ghimj(index,691);
        W_126 = W_126+ a *Ghimj(index,692);
        W_133 = W_133+ a *Ghimj(index,693);
        W_135 = W_135+ a *Ghimj(index,694);
        W_136 = W_136+ a *Ghimj(index,695);
        W_137 = W_137+ a *Ghimj(index,696);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        Ghimj(index,831) = W_75;
        Ghimj(index,832) = W_95;
        Ghimj(index,833) = W_96;
        Ghimj(index,834) = W_97;
        Ghimj(index,835) = W_98;
        Ghimj(index,836) = W_103;
        Ghimj(index,837) = W_106;
        Ghimj(index,838) = W_107;
        Ghimj(index,839) = W_108;
        Ghimj(index,840) = W_109;
        Ghimj(index,841) = W_110;
        Ghimj(index,842) = W_113;
        Ghimj(index,843) = W_115;
        Ghimj(index,844) = W_119;
        Ghimj(index,845) = W_120;
        Ghimj(index,846) = W_121;
        Ghimj(index,847) = W_122;
        Ghimj(index,848) = W_124;
        Ghimj(index,849) = W_125;
        Ghimj(index,850) = W_126;
        Ghimj(index,851) = W_127;
        Ghimj(index,852) = W_128;
        Ghimj(index,853) = W_129;
        Ghimj(index,854) = W_130;
        Ghimj(index,855) = W_131;
        Ghimj(index,856) = W_133;
        Ghimj(index,857) = W_135;
        Ghimj(index,858) = W_136;
        Ghimj(index,859) = W_137;
        Ghimj(index,860) = W_138;
        W_103 = Ghimj(index,861);
        W_104 = Ghimj(index,862);
        W_112 = Ghimj(index,863);
        W_114 = Ghimj(index,864);
        W_116 = Ghimj(index,865);
        W_118 = Ghimj(index,866);
        W_119 = Ghimj(index,867);
        W_121 = Ghimj(index,868);
        W_123 = Ghimj(index,869);
        W_124 = Ghimj(index,870);
        W_125 = Ghimj(index,871);
        W_126 = Ghimj(index,872);
        W_127 = Ghimj(index,873);
        W_128 = Ghimj(index,874);
        W_129 = Ghimj(index,875);
        W_130 = Ghimj(index,876);
        W_131 = Ghimj(index,877);
        W_132 = Ghimj(index,878);
        W_133 = Ghimj(index,879);
        W_134 = Ghimj(index,880);
        W_135 = Ghimj(index,881);
        W_136 = Ghimj(index,882);
        W_137 = Ghimj(index,883);
        W_138 = Ghimj(index,884);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_112/ Ghimj(index,677);
        W_112 = -a;
        W_116 = W_116+ a *Ghimj(index,678);
        W_123 = W_123+ a *Ghimj(index,679);
        W_126 = W_126+ a *Ghimj(index,680);
        W_128 = W_128+ a *Ghimj(index,681);
        W_134 = W_134+ a *Ghimj(index,682);
        W_137 = W_137+ a *Ghimj(index,683);
        W_138 = W_138+ a *Ghimj(index,684);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_118/ Ghimj(index,745);
        W_118 = -a;
        W_123 = W_123+ a *Ghimj(index,746);
        W_125 = W_125+ a *Ghimj(index,747);
        W_126 = W_126+ a *Ghimj(index,748);
        W_127 = W_127+ a *Ghimj(index,749);
        W_128 = W_128+ a *Ghimj(index,750);
        W_129 = W_129+ a *Ghimj(index,751);
        W_131 = W_131+ a *Ghimj(index,752);
        W_132 = W_132+ a *Ghimj(index,753);
        W_134 = W_134+ a *Ghimj(index,754);
        W_135 = W_135+ a *Ghimj(index,755);
        W_137 = W_137+ a *Ghimj(index,756);
        W_138 = W_138+ a *Ghimj(index,757);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        Ghimj(index,861) = W_103;
        Ghimj(index,862) = W_104;
        Ghimj(index,863) = W_112;
        Ghimj(index,864) = W_114;
        Ghimj(index,865) = W_116;
        Ghimj(index,866) = W_118;
        Ghimj(index,867) = W_119;
        Ghimj(index,868) = W_121;
        Ghimj(index,869) = W_123;
        Ghimj(index,870) = W_124;
        Ghimj(index,871) = W_125;
        Ghimj(index,872) = W_126;
        Ghimj(index,873) = W_127;
        Ghimj(index,874) = W_128;
        Ghimj(index,875) = W_129;
        Ghimj(index,876) = W_130;
        Ghimj(index,877) = W_131;
        Ghimj(index,878) = W_132;
        Ghimj(index,879) = W_133;
        Ghimj(index,880) = W_134;
        Ghimj(index,881) = W_135;
        Ghimj(index,882) = W_136;
        Ghimj(index,883) = W_137;
        Ghimj(index,884) = W_138;
        W_81 = Ghimj(index,885);
        W_84 = Ghimj(index,886);
        W_92 = Ghimj(index,887);
        W_103 = Ghimj(index,888);
        W_106 = Ghimj(index,889);
        W_107 = Ghimj(index,890);
        W_110 = Ghimj(index,891);
        W_114 = Ghimj(index,892);
        W_120 = Ghimj(index,893);
        W_121 = Ghimj(index,894);
        W_122 = Ghimj(index,895);
        W_124 = Ghimj(index,896);
        W_125 = Ghimj(index,897);
        W_126 = Ghimj(index,898);
        W_127 = Ghimj(index,899);
        W_128 = Ghimj(index,900);
        W_129 = Ghimj(index,901);
        W_130 = Ghimj(index,902);
        W_131 = Ghimj(index,903);
        W_132 = Ghimj(index,904);
        W_133 = Ghimj(index,905);
        W_135 = Ghimj(index,906);
        W_136 = Ghimj(index,907);
        W_137 = Ghimj(index,908);
        W_138 = Ghimj(index,909);
        a = - W_81/ Ghimj(index,405);
        W_81 = -a;
        W_114 = W_114+ a *Ghimj(index,406);
        W_124 = W_124+ a *Ghimj(index,407);
        W_126 = W_126+ a *Ghimj(index,408);
        W_127 = W_127+ a *Ghimj(index,409);
        W_129 = W_129+ a *Ghimj(index,410);
        W_136 = W_136+ a *Ghimj(index,411);
        a = - W_84/ Ghimj(index,421);
        W_84 = -a;
        W_92 = W_92+ a *Ghimj(index,422);
        W_124 = W_124+ a *Ghimj(index,423);
        W_135 = W_135+ a *Ghimj(index,424);
        W_137 = W_137+ a *Ghimj(index,425);
        a = - W_92/ Ghimj(index,489);
        W_92 = -a;
        W_124 = W_124+ a *Ghimj(index,490);
        W_126 = W_126+ a *Ghimj(index,491);
        W_133 = W_133+ a *Ghimj(index,492);
        W_135 = W_135+ a *Ghimj(index,493);
        W_137 = W_137+ a *Ghimj(index,494);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_110/ Ghimj(index,659);
        W_110 = -a;
        W_124 = W_124+ a *Ghimj(index,660);
        W_125 = W_125+ a *Ghimj(index,661);
        W_126 = W_126+ a *Ghimj(index,662);
        W_133 = W_133+ a *Ghimj(index,663);
        W_136 = W_136+ a *Ghimj(index,664);
        W_137 = W_137+ a *Ghimj(index,665);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        Ghimj(index,885) = W_81;
        Ghimj(index,886) = W_84;
        Ghimj(index,887) = W_92;
        Ghimj(index,888) = W_103;
        Ghimj(index,889) = W_106;
        Ghimj(index,890) = W_107;
        Ghimj(index,891) = W_110;
        Ghimj(index,892) = W_114;
        Ghimj(index,893) = W_120;
        Ghimj(index,894) = W_121;
        Ghimj(index,895) = W_122;
        Ghimj(index,896) = W_124;
        Ghimj(index,897) = W_125;
        Ghimj(index,898) = W_126;
        Ghimj(index,899) = W_127;
        Ghimj(index,900) = W_128;
        Ghimj(index,901) = W_129;
        Ghimj(index,902) = W_130;
        Ghimj(index,903) = W_131;
        Ghimj(index,904) = W_132;
        Ghimj(index,905) = W_133;
        Ghimj(index,906) = W_135;
        Ghimj(index,907) = W_136;
        Ghimj(index,908) = W_137;
        Ghimj(index,909) = W_138;
        W_3 = Ghimj(index,910);
        W_53 = Ghimj(index,911);
        W_63 = Ghimj(index,912);
        W_65 = Ghimj(index,913);
        W_74 = Ghimj(index,914);
        W_75 = Ghimj(index,915);
        W_81 = Ghimj(index,916);
        W_86 = Ghimj(index,917);
        W_93 = Ghimj(index,918);
        W_94 = Ghimj(index,919);
        W_98 = Ghimj(index,920);
        W_102 = Ghimj(index,921);
        W_104 = Ghimj(index,922);
        W_106 = Ghimj(index,923);
        W_107 = Ghimj(index,924);
        W_109 = Ghimj(index,925);
        W_113 = Ghimj(index,926);
        W_114 = Ghimj(index,927);
        W_117 = Ghimj(index,928);
        W_119 = Ghimj(index,929);
        W_120 = Ghimj(index,930);
        W_121 = Ghimj(index,931);
        W_122 = Ghimj(index,932);
        W_124 = Ghimj(index,933);
        W_125 = Ghimj(index,934);
        W_126 = Ghimj(index,935);
        W_127 = Ghimj(index,936);
        W_128 = Ghimj(index,937);
        W_129 = Ghimj(index,938);
        W_130 = Ghimj(index,939);
        W_131 = Ghimj(index,940);
        W_132 = Ghimj(index,941);
        W_133 = Ghimj(index,942);
        W_134 = Ghimj(index,943);
        W_135 = Ghimj(index,944);
        W_136 = Ghimj(index,945);
        W_137 = Ghimj(index,946);
        W_138 = Ghimj(index,947);
        a = - W_3/ Ghimj(index,3);
        W_3 = -a;
        a = - W_53/ Ghimj(index,290);
        W_53 = -a;
        W_126 = W_126+ a *Ghimj(index,291);
        a = - W_63/ Ghimj(index,323);
        W_63 = -a;
        W_121 = W_121+ a *Ghimj(index,324);
        W_126 = W_126+ a *Ghimj(index,325);
        W_137 = W_137+ a *Ghimj(index,326);
        a = - W_65/ Ghimj(index,331);
        W_65 = -a;
        W_114 = W_114+ a *Ghimj(index,332);
        W_126 = W_126+ a *Ghimj(index,333);
        W_132 = W_132+ a *Ghimj(index,334);
        a = - W_74/ Ghimj(index,368);
        W_74 = -a;
        W_117 = W_117+ a *Ghimj(index,369);
        W_121 = W_121+ a *Ghimj(index,370);
        W_125 = W_125+ a *Ghimj(index,371);
        W_126 = W_126+ a *Ghimj(index,372);
        W_137 = W_137+ a *Ghimj(index,373);
        a = - W_75/ Ghimj(index,374);
        W_75 = -a;
        W_120 = W_120+ a *Ghimj(index,375);
        W_126 = W_126+ a *Ghimj(index,376);
        a = - W_81/ Ghimj(index,405);
        W_81 = -a;
        W_114 = W_114+ a *Ghimj(index,406);
        W_124 = W_124+ a *Ghimj(index,407);
        W_126 = W_126+ a *Ghimj(index,408);
        W_127 = W_127+ a *Ghimj(index,409);
        W_129 = W_129+ a *Ghimj(index,410);
        W_136 = W_136+ a *Ghimj(index,411);
        a = - W_86/ Ghimj(index,436);
        W_86 = -a;
        W_93 = W_93+ a *Ghimj(index,437);
        W_125 = W_125+ a *Ghimj(index,438);
        W_126 = W_126+ a *Ghimj(index,439);
        W_133 = W_133+ a *Ghimj(index,440);
        W_137 = W_137+ a *Ghimj(index,441);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        a = - W_98/ Ghimj(index,557);
        W_98 = -a;
        W_107 = W_107+ a *Ghimj(index,558);
        W_120 = W_120+ a *Ghimj(index,559);
        W_124 = W_124+ a *Ghimj(index,560);
        W_126 = W_126+ a *Ghimj(index,561);
        W_127 = W_127+ a *Ghimj(index,562);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        a = - W_113/ Ghimj(index,689);
        W_113 = -a;
        W_124 = W_124+ a *Ghimj(index,690);
        W_125 = W_125+ a *Ghimj(index,691);
        W_126 = W_126+ a *Ghimj(index,692);
        W_133 = W_133+ a *Ghimj(index,693);
        W_135 = W_135+ a *Ghimj(index,694);
        W_136 = W_136+ a *Ghimj(index,695);
        W_137 = W_137+ a *Ghimj(index,696);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        Ghimj(index,910) = W_3;
        Ghimj(index,911) = W_53;
        Ghimj(index,912) = W_63;
        Ghimj(index,913) = W_65;
        Ghimj(index,914) = W_74;
        Ghimj(index,915) = W_75;
        Ghimj(index,916) = W_81;
        Ghimj(index,917) = W_86;
        Ghimj(index,918) = W_93;
        Ghimj(index,919) = W_94;
        Ghimj(index,920) = W_98;
        Ghimj(index,921) = W_102;
        Ghimj(index,922) = W_104;
        Ghimj(index,923) = W_106;
        Ghimj(index,924) = W_107;
        Ghimj(index,925) = W_109;
        Ghimj(index,926) = W_113;
        Ghimj(index,927) = W_114;
        Ghimj(index,928) = W_117;
        Ghimj(index,929) = W_119;
        Ghimj(index,930) = W_120;
        Ghimj(index,931) = W_121;
        Ghimj(index,932) = W_122;
        Ghimj(index,933) = W_124;
        Ghimj(index,934) = W_125;
        Ghimj(index,935) = W_126;
        Ghimj(index,936) = W_127;
        Ghimj(index,937) = W_128;
        Ghimj(index,938) = W_129;
        Ghimj(index,939) = W_130;
        Ghimj(index,940) = W_131;
        Ghimj(index,941) = W_132;
        Ghimj(index,942) = W_133;
        Ghimj(index,943) = W_134;
        Ghimj(index,944) = W_135;
        Ghimj(index,945) = W_136;
        Ghimj(index,946) = W_137;
        Ghimj(index,947) = W_138;
        W_40 = Ghimj(index,948);
        W_44 = Ghimj(index,949);
        W_45 = Ghimj(index,950);
        W_47 = Ghimj(index,951);
        W_48 = Ghimj(index,952);
        W_49 = Ghimj(index,953);
        W_52 = Ghimj(index,954);
        W_53 = Ghimj(index,955);
        W_54 = Ghimj(index,956);
        W_55 = Ghimj(index,957);
        W_56 = Ghimj(index,958);
        W_57 = Ghimj(index,959);
        W_58 = Ghimj(index,960);
        W_61 = Ghimj(index,961);
        W_62 = Ghimj(index,962);
        W_63 = Ghimj(index,963);
        W_64 = Ghimj(index,964);
        W_65 = Ghimj(index,965);
        W_66 = Ghimj(index,966);
        W_67 = Ghimj(index,967);
        W_68 = Ghimj(index,968);
        W_69 = Ghimj(index,969);
        W_70 = Ghimj(index,970);
        W_71 = Ghimj(index,971);
        W_72 = Ghimj(index,972);
        W_73 = Ghimj(index,973);
        W_74 = Ghimj(index,974);
        W_75 = Ghimj(index,975);
        W_76 = Ghimj(index,976);
        W_77 = Ghimj(index,977);
        W_78 = Ghimj(index,978);
        W_79 = Ghimj(index,979);
        W_81 = Ghimj(index,980);
        W_82 = Ghimj(index,981);
        W_84 = Ghimj(index,982);
        W_85 = Ghimj(index,983);
        W_86 = Ghimj(index,984);
        W_87 = Ghimj(index,985);
        W_88 = Ghimj(index,986);
        W_89 = Ghimj(index,987);
        W_91 = Ghimj(index,988);
        W_92 = Ghimj(index,989);
        W_93 = Ghimj(index,990);
        W_94 = Ghimj(index,991);
        W_95 = Ghimj(index,992);
        W_96 = Ghimj(index,993);
        W_97 = Ghimj(index,994);
        W_98 = Ghimj(index,995);
        W_99 = Ghimj(index,996);
        W_100 = Ghimj(index,997);
        W_101 = Ghimj(index,998);
        W_102 = Ghimj(index,999);
        W_103 = Ghimj(index,1000);
        W_104 = Ghimj(index,1001);
        W_105 = Ghimj(index,1002);
        W_106 = Ghimj(index,1003);
        W_107 = Ghimj(index,1004);
        W_108 = Ghimj(index,1005);
        W_109 = Ghimj(index,1006);
        W_110 = Ghimj(index,1007);
        W_111 = Ghimj(index,1008);
        W_112 = Ghimj(index,1009);
        W_113 = Ghimj(index,1010);
        W_114 = Ghimj(index,1011);
        W_115 = Ghimj(index,1012);
        W_116 = Ghimj(index,1013);
        W_117 = Ghimj(index,1014);
        W_118 = Ghimj(index,1015);
        W_119 = Ghimj(index,1016);
        W_120 = Ghimj(index,1017);
        W_121 = Ghimj(index,1018);
        W_122 = Ghimj(index,1019);
        W_123 = Ghimj(index,1020);
        W_124 = Ghimj(index,1021);
        W_125 = Ghimj(index,1022);
        W_126 = Ghimj(index,1023);
        W_127 = Ghimj(index,1024);
        W_128 = Ghimj(index,1025);
        W_129 = Ghimj(index,1026);
        W_130 = Ghimj(index,1027);
        W_131 = Ghimj(index,1028);
        W_132 = Ghimj(index,1029);
        W_133 = Ghimj(index,1030);
        W_134 = Ghimj(index,1031);
        W_135 = Ghimj(index,1032);
        W_136 = Ghimj(index,1033);
        W_137 = Ghimj(index,1034);
        W_138 = Ghimj(index,1035);
        a = - W_40/ Ghimj(index,260);
        W_40 = -a;
        W_126 = W_126+ a *Ghimj(index,261);
        a = - W_44/ Ghimj(index,268);
        W_44 = -a;
        W_126 = W_126+ a *Ghimj(index,269);
        a = - W_45/ Ghimj(index,270);
        W_45 = -a;
        W_126 = W_126+ a *Ghimj(index,271);
        a = - W_47/ Ghimj(index,276);
        W_47 = -a;
        W_126 = W_126+ a *Ghimj(index,277);
        a = - W_48/ Ghimj(index,278);
        W_48 = -a;
        W_126 = W_126+ a *Ghimj(index,279);
        a = - W_49/ Ghimj(index,280);
        W_49 = -a;
        W_126 = W_126+ a *Ghimj(index,281);
        a = - W_52/ Ghimj(index,288);
        W_52 = -a;
        W_126 = W_126+ a *Ghimj(index,289);
        a = - W_53/ Ghimj(index,290);
        W_53 = -a;
        W_126 = W_126+ a *Ghimj(index,291);
        a = - W_54/ Ghimj(index,292);
        W_54 = -a;
        W_126 = W_126+ a *Ghimj(index,293);
        a = - W_55/ Ghimj(index,294);
        W_55 = -a;
        W_126 = W_126+ a *Ghimj(index,295);
        a = - W_56/ Ghimj(index,296);
        W_56 = -a;
        W_65 = W_65+ a *Ghimj(index,297);
        W_81 = W_81+ a *Ghimj(index,298);
        W_126 = W_126+ a *Ghimj(index,299);
        a = - W_57/ Ghimj(index,300);
        W_57 = -a;
        W_120 = W_120+ a *Ghimj(index,301);
        W_126 = W_126+ a *Ghimj(index,302);
        a = - W_58/ Ghimj(index,303);
        W_58 = -a;
        W_91 = W_91+ a *Ghimj(index,304);
        W_126 = W_126+ a *Ghimj(index,305);
        a = - W_61/ Ghimj(index,315);
        W_61 = -a;
        W_70 = W_70+ a *Ghimj(index,316);
        W_87 = W_87+ a *Ghimj(index,317);
        W_126 = W_126+ a *Ghimj(index,318);
        a = - W_62/ Ghimj(index,319);
        W_62 = -a;
        W_93 = W_93+ a *Ghimj(index,320);
        W_126 = W_126+ a *Ghimj(index,321);
        W_133 = W_133+ a *Ghimj(index,322);
        a = - W_63/ Ghimj(index,323);
        W_63 = -a;
        W_121 = W_121+ a *Ghimj(index,324);
        W_126 = W_126+ a *Ghimj(index,325);
        W_137 = W_137+ a *Ghimj(index,326);
        a = - W_64/ Ghimj(index,327);
        W_64 = -a;
        W_113 = W_113+ a *Ghimj(index,328);
        W_126 = W_126+ a *Ghimj(index,329);
        W_135 = W_135+ a *Ghimj(index,330);
        a = - W_65/ Ghimj(index,331);
        W_65 = -a;
        W_114 = W_114+ a *Ghimj(index,332);
        W_126 = W_126+ a *Ghimj(index,333);
        W_132 = W_132+ a *Ghimj(index,334);
        a = - W_66/ Ghimj(index,335);
        W_66 = -a;
        W_109 = W_109+ a *Ghimj(index,336);
        W_126 = W_126+ a *Ghimj(index,337);
        W_137 = W_137+ a *Ghimj(index,338);
        a = - W_67/ Ghimj(index,339);
        W_67 = -a;
        W_115 = W_115+ a *Ghimj(index,340);
        W_126 = W_126+ a *Ghimj(index,341);
        W_137 = W_137+ a *Ghimj(index,342);
        a = - W_68/ Ghimj(index,343);
        W_68 = -a;
        W_99 = W_99+ a *Ghimj(index,344);
        W_126 = W_126+ a *Ghimj(index,345);
        W_137 = W_137+ a *Ghimj(index,346);
        a = - W_69/ Ghimj(index,347);
        W_69 = -a;
        W_93 = W_93+ a *Ghimj(index,348);
        W_126 = W_126+ a *Ghimj(index,349);
        W_137 = W_137+ a *Ghimj(index,350);
        a = - W_70/ Ghimj(index,352);
        W_70 = -a;
        W_84 = W_84+ a *Ghimj(index,353);
        W_87 = W_87+ a *Ghimj(index,354);
        W_126 = W_126+ a *Ghimj(index,355);
        a = - W_71/ Ghimj(index,356);
        W_71 = -a;
        W_117 = W_117+ a *Ghimj(index,357);
        W_126 = W_126+ a *Ghimj(index,358);
        W_137 = W_137+ a *Ghimj(index,359);
        a = - W_72/ Ghimj(index,360);
        W_72 = -a;
        W_94 = W_94+ a *Ghimj(index,361);
        W_126 = W_126+ a *Ghimj(index,362);
        W_137 = W_137+ a *Ghimj(index,363);
        a = - W_73/ Ghimj(index,364);
        W_73 = -a;
        W_126 = W_126+ a *Ghimj(index,365);
        W_135 = W_135+ a *Ghimj(index,366);
        W_137 = W_137+ a *Ghimj(index,367);
        a = - W_74/ Ghimj(index,368);
        W_74 = -a;
        W_117 = W_117+ a *Ghimj(index,369);
        W_121 = W_121+ a *Ghimj(index,370);
        W_125 = W_125+ a *Ghimj(index,371);
        W_126 = W_126+ a *Ghimj(index,372);
        W_137 = W_137+ a *Ghimj(index,373);
        a = - W_75/ Ghimj(index,374);
        W_75 = -a;
        W_120 = W_120+ a *Ghimj(index,375);
        W_126 = W_126+ a *Ghimj(index,376);
        a = - W_76/ Ghimj(index,377);
        W_76 = -a;
        W_87 = W_87+ a *Ghimj(index,378);
        W_126 = W_126+ a *Ghimj(index,379);
        W_133 = W_133+ a *Ghimj(index,380);
        W_135 = W_135+ a *Ghimj(index,381);
        a = - W_77/ Ghimj(index,382);
        W_77 = -a;
        W_121 = W_121+ a *Ghimj(index,383);
        W_126 = W_126+ a *Ghimj(index,384);
        W_135 = W_135+ a *Ghimj(index,385);
        a = - W_78/ Ghimj(index,386);
        W_78 = -a;
        W_103 = W_103+ a *Ghimj(index,387);
        W_106 = W_106+ a *Ghimj(index,388);
        W_107 = W_107+ a *Ghimj(index,389);
        W_110 = W_110+ a *Ghimj(index,390);
        W_124 = W_124+ a *Ghimj(index,391);
        W_126 = W_126+ a *Ghimj(index,392);
        a = - W_79/ Ghimj(index,393);
        W_79 = -a;
        W_102 = W_102+ a *Ghimj(index,394);
        W_126 = W_126+ a *Ghimj(index,395);
        W_137 = W_137+ a *Ghimj(index,396);
        a = - W_81/ Ghimj(index,405);
        W_81 = -a;
        W_114 = W_114+ a *Ghimj(index,406);
        W_124 = W_124+ a *Ghimj(index,407);
        W_126 = W_126+ a *Ghimj(index,408);
        W_127 = W_127+ a *Ghimj(index,409);
        W_129 = W_129+ a *Ghimj(index,410);
        W_136 = W_136+ a *Ghimj(index,411);
        a = - W_82/ Ghimj(index,412);
        W_82 = -a;
        W_113 = W_113+ a *Ghimj(index,413);
        W_126 = W_126+ a *Ghimj(index,414);
        W_137 = W_137+ a *Ghimj(index,415);
        a = - W_84/ Ghimj(index,421);
        W_84 = -a;
        W_92 = W_92+ a *Ghimj(index,422);
        W_124 = W_124+ a *Ghimj(index,423);
        W_135 = W_135+ a *Ghimj(index,424);
        W_137 = W_137+ a *Ghimj(index,425);
        a = - W_85/ Ghimj(index,427);
        W_85 = -a;
        W_102 = W_102+ a *Ghimj(index,428);
        W_111 = W_111+ a *Ghimj(index,429);
        W_125 = W_125+ a *Ghimj(index,430);
        W_126 = W_126+ a *Ghimj(index,431);
        W_133 = W_133+ a *Ghimj(index,432);
        W_137 = W_137+ a *Ghimj(index,433);
        a = - W_86/ Ghimj(index,436);
        W_86 = -a;
        W_93 = W_93+ a *Ghimj(index,437);
        W_125 = W_125+ a *Ghimj(index,438);
        W_126 = W_126+ a *Ghimj(index,439);
        W_133 = W_133+ a *Ghimj(index,440);
        W_137 = W_137+ a *Ghimj(index,441);
        a = - W_87/ Ghimj(index,444);
        W_87 = -a;
        W_92 = W_92+ a *Ghimj(index,445);
        W_124 = W_124+ a *Ghimj(index,446);
        W_126 = W_126+ a *Ghimj(index,447);
        W_135 = W_135+ a *Ghimj(index,448);
        W_137 = W_137+ a *Ghimj(index,449);
        a = - W_88/ Ghimj(index,450);
        W_88 = -a;
        W_103 = W_103+ a *Ghimj(index,451);
        W_106 = W_106+ a *Ghimj(index,452);
        W_124 = W_124+ a *Ghimj(index,453);
        W_126 = W_126+ a *Ghimj(index,454);
        W_127 = W_127+ a *Ghimj(index,455);
        W_137 = W_137+ a *Ghimj(index,456);
        a = - W_89/ Ghimj(index,457);
        W_89 = -a;
        W_93 = W_93+ a *Ghimj(index,458);
        W_94 = W_94+ a *Ghimj(index,459);
        W_102 = W_102+ a *Ghimj(index,460);
        W_107 = W_107+ a *Ghimj(index,461);
        W_109 = W_109+ a *Ghimj(index,462);
        W_113 = W_113+ a *Ghimj(index,463);
        W_117 = W_117+ a *Ghimj(index,464);
        W_124 = W_124+ a *Ghimj(index,465);
        W_125 = W_125+ a *Ghimj(index,466);
        W_126 = W_126+ a *Ghimj(index,467);
        a = - W_91/ Ghimj(index,481);
        W_91 = -a;
        W_106 = W_106+ a *Ghimj(index,482);
        W_109 = W_109+ a *Ghimj(index,483);
        W_126 = W_126+ a *Ghimj(index,484);
        W_133 = W_133+ a *Ghimj(index,485);
        W_136 = W_136+ a *Ghimj(index,486);
        a = - W_92/ Ghimj(index,489);
        W_92 = -a;
        W_124 = W_124+ a *Ghimj(index,490);
        W_126 = W_126+ a *Ghimj(index,491);
        W_133 = W_133+ a *Ghimj(index,492);
        W_135 = W_135+ a *Ghimj(index,493);
        W_137 = W_137+ a *Ghimj(index,494);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        a = - W_95/ Ghimj(index,514);
        W_95 = -a;
        W_96 = W_96+ a *Ghimj(index,515);
        W_98 = W_98+ a *Ghimj(index,516);
        W_103 = W_103+ a *Ghimj(index,517);
        W_106 = W_106+ a *Ghimj(index,518);
        W_107 = W_107+ a *Ghimj(index,519);
        W_109 = W_109+ a *Ghimj(index,520);
        W_110 = W_110+ a *Ghimj(index,521);
        W_113 = W_113+ a *Ghimj(index,522);
        W_119 = W_119+ a *Ghimj(index,523);
        W_121 = W_121+ a *Ghimj(index,524);
        W_124 = W_124+ a *Ghimj(index,525);
        W_125 = W_125+ a *Ghimj(index,526);
        W_126 = W_126+ a *Ghimj(index,527);
        W_127 = W_127+ a *Ghimj(index,528);
        W_129 = W_129+ a *Ghimj(index,529);
        W_130 = W_130+ a *Ghimj(index,530);
        W_133 = W_133+ a *Ghimj(index,531);
        W_135 = W_135+ a *Ghimj(index,532);
        W_136 = W_136+ a *Ghimj(index,533);
        W_137 = W_137+ a *Ghimj(index,534);
        a = - W_96/ Ghimj(index,538);
        W_96 = -a;
        W_107 = W_107+ a *Ghimj(index,539);
        W_108 = W_108+ a *Ghimj(index,540);
        W_109 = W_109+ a *Ghimj(index,541);
        W_110 = W_110+ a *Ghimj(index,542);
        W_113 = W_113+ a *Ghimj(index,543);
        W_124 = W_124+ a *Ghimj(index,544);
        W_125 = W_125+ a *Ghimj(index,545);
        W_126 = W_126+ a *Ghimj(index,546);
        W_133 = W_133+ a *Ghimj(index,547);
        W_137 = W_137+ a *Ghimj(index,548);
        a = - W_97/ Ghimj(index,549);
        W_97 = -a;
        W_98 = W_98+ a *Ghimj(index,550);
        W_120 = W_120+ a *Ghimj(index,551);
        W_122 = W_122+ a *Ghimj(index,552);
        W_126 = W_126+ a *Ghimj(index,553);
        W_127 = W_127+ a *Ghimj(index,554);
        W_130 = W_130+ a *Ghimj(index,555);
        W_137 = W_137+ a *Ghimj(index,556);
        a = - W_98/ Ghimj(index,557);
        W_98 = -a;
        W_107 = W_107+ a *Ghimj(index,558);
        W_120 = W_120+ a *Ghimj(index,559);
        W_124 = W_124+ a *Ghimj(index,560);
        W_126 = W_126+ a *Ghimj(index,561);
        W_127 = W_127+ a *Ghimj(index,562);
        a = - W_99/ Ghimj(index,565);
        W_99 = -a;
        W_102 = W_102+ a *Ghimj(index,566);
        W_111 = W_111+ a *Ghimj(index,567);
        W_125 = W_125+ a *Ghimj(index,568);
        W_126 = W_126+ a *Ghimj(index,569);
        W_133 = W_133+ a *Ghimj(index,570);
        W_137 = W_137+ a *Ghimj(index,571);
        a = - W_100/ Ghimj(index,573);
        W_100 = -a;
        W_105 = W_105+ a *Ghimj(index,574);
        W_112 = W_112+ a *Ghimj(index,575);
        W_116 = W_116+ a *Ghimj(index,576);
        W_118 = W_118+ a *Ghimj(index,577);
        W_123 = W_123+ a *Ghimj(index,578);
        W_126 = W_126+ a *Ghimj(index,579);
        W_127 = W_127+ a *Ghimj(index,580);
        W_129 = W_129+ a *Ghimj(index,581);
        W_132 = W_132+ a *Ghimj(index,582);
        W_134 = W_134+ a *Ghimj(index,583);
        W_138 = W_138+ a *Ghimj(index,584);
        a = - W_101/ Ghimj(index,586);
        W_101 = -a;
        W_105 = W_105+ a *Ghimj(index,587);
        W_114 = W_114+ a *Ghimj(index,588);
        W_116 = W_116+ a *Ghimj(index,589);
        W_119 = W_119+ a *Ghimj(index,590);
        W_123 = W_123+ a *Ghimj(index,591);
        W_126 = W_126+ a *Ghimj(index,592);
        W_128 = W_128+ a *Ghimj(index,593);
        W_130 = W_130+ a *Ghimj(index,594);
        W_135 = W_135+ a *Ghimj(index,595);
        W_136 = W_136+ a *Ghimj(index,596);
        W_138 = W_138+ a *Ghimj(index,597);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_108/ Ghimj(index,636);
        W_108 = -a;
        W_109 = W_109+ a *Ghimj(index,637);
        W_113 = W_113+ a *Ghimj(index,638);
        W_115 = W_115+ a *Ghimj(index,639);
        W_124 = W_124+ a *Ghimj(index,640);
        W_125 = W_125+ a *Ghimj(index,641);
        W_126 = W_126+ a *Ghimj(index,642);
        W_133 = W_133+ a *Ghimj(index,643);
        W_135 = W_135+ a *Ghimj(index,644);
        W_136 = W_136+ a *Ghimj(index,645);
        W_137 = W_137+ a *Ghimj(index,646);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        a = - W_110/ Ghimj(index,659);
        W_110 = -a;
        W_124 = W_124+ a *Ghimj(index,660);
        W_125 = W_125+ a *Ghimj(index,661);
        W_126 = W_126+ a *Ghimj(index,662);
        W_133 = W_133+ a *Ghimj(index,663);
        W_136 = W_136+ a *Ghimj(index,664);
        W_137 = W_137+ a *Ghimj(index,665);
        a = - W_111/ Ghimj(index,669);
        W_111 = -a;
        W_115 = W_115+ a *Ghimj(index,670);
        W_124 = W_124+ a *Ghimj(index,671);
        W_125 = W_125+ a *Ghimj(index,672);
        W_126 = W_126+ a *Ghimj(index,673);
        W_133 = W_133+ a *Ghimj(index,674);
        W_136 = W_136+ a *Ghimj(index,675);
        W_137 = W_137+ a *Ghimj(index,676);
        a = - W_112/ Ghimj(index,677);
        W_112 = -a;
        W_116 = W_116+ a *Ghimj(index,678);
        W_123 = W_123+ a *Ghimj(index,679);
        W_126 = W_126+ a *Ghimj(index,680);
        W_128 = W_128+ a *Ghimj(index,681);
        W_134 = W_134+ a *Ghimj(index,682);
        W_137 = W_137+ a *Ghimj(index,683);
        W_138 = W_138+ a *Ghimj(index,684);
        a = - W_113/ Ghimj(index,689);
        W_113 = -a;
        W_124 = W_124+ a *Ghimj(index,690);
        W_125 = W_125+ a *Ghimj(index,691);
        W_126 = W_126+ a *Ghimj(index,692);
        W_133 = W_133+ a *Ghimj(index,693);
        W_135 = W_135+ a *Ghimj(index,694);
        W_136 = W_136+ a *Ghimj(index,695);
        W_137 = W_137+ a *Ghimj(index,696);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_118/ Ghimj(index,745);
        W_118 = -a;
        W_123 = W_123+ a *Ghimj(index,746);
        W_125 = W_125+ a *Ghimj(index,747);
        W_126 = W_126+ a *Ghimj(index,748);
        W_127 = W_127+ a *Ghimj(index,749);
        W_128 = W_128+ a *Ghimj(index,750);
        W_129 = W_129+ a *Ghimj(index,751);
        W_131 = W_131+ a *Ghimj(index,752);
        W_132 = W_132+ a *Ghimj(index,753);
        W_134 = W_134+ a *Ghimj(index,754);
        W_135 = W_135+ a *Ghimj(index,755);
        W_137 = W_137+ a *Ghimj(index,756);
        W_138 = W_138+ a *Ghimj(index,757);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        Ghimj(index,948) = W_40;
        Ghimj(index,949) = W_44;
        Ghimj(index,950) = W_45;
        Ghimj(index,951) = W_47;
        Ghimj(index,952) = W_48;
        Ghimj(index,953) = W_49;
        Ghimj(index,954) = W_52;
        Ghimj(index,955) = W_53;
        Ghimj(index,956) = W_54;
        Ghimj(index,957) = W_55;
        Ghimj(index,958) = W_56;
        Ghimj(index,959) = W_57;
        Ghimj(index,960) = W_58;
        Ghimj(index,961) = W_61;
        Ghimj(index,962) = W_62;
        Ghimj(index,963) = W_63;
        Ghimj(index,964) = W_64;
        Ghimj(index,965) = W_65;
        Ghimj(index,966) = W_66;
        Ghimj(index,967) = W_67;
        Ghimj(index,968) = W_68;
        Ghimj(index,969) = W_69;
        Ghimj(index,970) = W_70;
        Ghimj(index,971) = W_71;
        Ghimj(index,972) = W_72;
        Ghimj(index,973) = W_73;
        Ghimj(index,974) = W_74;
        Ghimj(index,975) = W_75;
        Ghimj(index,976) = W_76;
        Ghimj(index,977) = W_77;
        Ghimj(index,978) = W_78;
        Ghimj(index,979) = W_79;
        Ghimj(index,980) = W_81;
        Ghimj(index,981) = W_82;
        Ghimj(index,982) = W_84;
        Ghimj(index,983) = W_85;
        Ghimj(index,984) = W_86;
        Ghimj(index,985) = W_87;
        Ghimj(index,986) = W_88;
        Ghimj(index,987) = W_89;
        Ghimj(index,988) = W_91;
        Ghimj(index,989) = W_92;
        Ghimj(index,990) = W_93;
        Ghimj(index,991) = W_94;
        Ghimj(index,992) = W_95;
        Ghimj(index,993) = W_96;
        Ghimj(index,994) = W_97;
        Ghimj(index,995) = W_98;
        Ghimj(index,996) = W_99;
        Ghimj(index,997) = W_100;
        Ghimj(index,998) = W_101;
        Ghimj(index,999) = W_102;
        Ghimj(index,1000) = W_103;
        Ghimj(index,1001) = W_104;
        Ghimj(index,1002) = W_105;
        Ghimj(index,1003) = W_106;
        Ghimj(index,1004) = W_107;
        Ghimj(index,1005) = W_108;
        Ghimj(index,1006) = W_109;
        Ghimj(index,1007) = W_110;
        Ghimj(index,1008) = W_111;
        Ghimj(index,1009) = W_112;
        Ghimj(index,1010) = W_113;
        Ghimj(index,1011) = W_114;
        Ghimj(index,1012) = W_115;
        Ghimj(index,1013) = W_116;
        Ghimj(index,1014) = W_117;
        Ghimj(index,1015) = W_118;
        Ghimj(index,1016) = W_119;
        Ghimj(index,1017) = W_120;
        Ghimj(index,1018) = W_121;
        Ghimj(index,1019) = W_122;
        Ghimj(index,1020) = W_123;
        Ghimj(index,1021) = W_124;
        Ghimj(index,1022) = W_125;
        Ghimj(index,1023) = W_126;
        Ghimj(index,1024) = W_127;
        Ghimj(index,1025) = W_128;
        Ghimj(index,1026) = W_129;
        Ghimj(index,1027) = W_130;
        Ghimj(index,1028) = W_131;
        Ghimj(index,1029) = W_132;
        Ghimj(index,1030) = W_133;
        Ghimj(index,1031) = W_134;
        Ghimj(index,1032) = W_135;
        Ghimj(index,1033) = W_136;
        Ghimj(index,1034) = W_137;
        Ghimj(index,1035) = W_138;
        W_1 = Ghimj(index,1036);
        W_39 = Ghimj(index,1037);
        W_41 = Ghimj(index,1038);
        W_42 = Ghimj(index,1039);
        W_43 = Ghimj(index,1040);
        W_50 = Ghimj(index,1041);
        W_52 = Ghimj(index,1042);
        W_54 = Ghimj(index,1043);
        W_55 = Ghimj(index,1044);
        W_57 = Ghimj(index,1045);
        W_75 = Ghimj(index,1046);
        W_80 = Ghimj(index,1047);
        W_83 = Ghimj(index,1048);
        W_88 = Ghimj(index,1049);
        W_90 = Ghimj(index,1050);
        W_97 = Ghimj(index,1051);
        W_98 = Ghimj(index,1052);
        W_100 = Ghimj(index,1053);
        W_103 = Ghimj(index,1054);
        W_104 = Ghimj(index,1055);
        W_105 = Ghimj(index,1056);
        W_106 = Ghimj(index,1057);
        W_107 = Ghimj(index,1058);
        W_112 = Ghimj(index,1059);
        W_114 = Ghimj(index,1060);
        W_116 = Ghimj(index,1061);
        W_118 = Ghimj(index,1062);
        W_119 = Ghimj(index,1063);
        W_120 = Ghimj(index,1064);
        W_121 = Ghimj(index,1065);
        W_122 = Ghimj(index,1066);
        W_123 = Ghimj(index,1067);
        W_124 = Ghimj(index,1068);
        W_125 = Ghimj(index,1069);
        W_126 = Ghimj(index,1070);
        W_127 = Ghimj(index,1071);
        W_128 = Ghimj(index,1072);
        W_129 = Ghimj(index,1073);
        W_130 = Ghimj(index,1074);
        W_131 = Ghimj(index,1075);
        W_132 = Ghimj(index,1076);
        W_133 = Ghimj(index,1077);
        W_134 = Ghimj(index,1078);
        W_135 = Ghimj(index,1079);
        W_136 = Ghimj(index,1080);
        W_137 = Ghimj(index,1081);
        W_138 = Ghimj(index,1082);
        a = - W_1/ Ghimj(index,1);
        W_1 = -a;
        a = - W_39/ Ghimj(index,258);
        W_39 = -a;
        W_134 = W_134+ a *Ghimj(index,259);
        a = - W_41/ Ghimj(index,262);
        W_41 = -a;
        W_120 = W_120+ a *Ghimj(index,263);
        a = - W_42/ Ghimj(index,264);
        W_42 = -a;
        W_120 = W_120+ a *Ghimj(index,265);
        a = - W_43/ Ghimj(index,266);
        W_43 = -a;
        W_120 = W_120+ a *Ghimj(index,267);
        a = - W_50/ Ghimj(index,282);
        W_50 = -a;
        W_83 = W_83+ a *Ghimj(index,283);
        W_138 = W_138+ a *Ghimj(index,284);
        a = - W_52/ Ghimj(index,288);
        W_52 = -a;
        W_126 = W_126+ a *Ghimj(index,289);
        a = - W_54/ Ghimj(index,292);
        W_54 = -a;
        W_126 = W_126+ a *Ghimj(index,293);
        a = - W_55/ Ghimj(index,294);
        W_55 = -a;
        W_126 = W_126+ a *Ghimj(index,295);
        a = - W_57/ Ghimj(index,300);
        W_57 = -a;
        W_120 = W_120+ a *Ghimj(index,301);
        W_126 = W_126+ a *Ghimj(index,302);
        a = - W_75/ Ghimj(index,374);
        W_75 = -a;
        W_120 = W_120+ a *Ghimj(index,375);
        W_126 = W_126+ a *Ghimj(index,376);
        a = - W_80/ Ghimj(index,397);
        W_80 = -a;
        W_90 = W_90+ a *Ghimj(index,398);
        W_112 = W_112+ a *Ghimj(index,399);
        W_116 = W_116+ a *Ghimj(index,400);
        W_127 = W_127+ a *Ghimj(index,401);
        W_129 = W_129+ a *Ghimj(index,402);
        W_134 = W_134+ a *Ghimj(index,403);
        W_138 = W_138+ a *Ghimj(index,404);
        a = - W_83/ Ghimj(index,416);
        W_83 = -a;
        W_128 = W_128+ a *Ghimj(index,417);
        W_135 = W_135+ a *Ghimj(index,418);
        W_136 = W_136+ a *Ghimj(index,419);
        W_138 = W_138+ a *Ghimj(index,420);
        a = - W_88/ Ghimj(index,450);
        W_88 = -a;
        W_103 = W_103+ a *Ghimj(index,451);
        W_106 = W_106+ a *Ghimj(index,452);
        W_124 = W_124+ a *Ghimj(index,453);
        W_126 = W_126+ a *Ghimj(index,454);
        W_127 = W_127+ a *Ghimj(index,455);
        W_137 = W_137+ a *Ghimj(index,456);
        a = - W_90/ Ghimj(index,469);
        W_90 = -a;
        W_100 = W_100+ a *Ghimj(index,470);
        W_105 = W_105+ a *Ghimj(index,471);
        W_112 = W_112+ a *Ghimj(index,472);
        W_116 = W_116+ a *Ghimj(index,473);
        W_118 = W_118+ a *Ghimj(index,474);
        W_123 = W_123+ a *Ghimj(index,475);
        W_127 = W_127+ a *Ghimj(index,476);
        W_129 = W_129+ a *Ghimj(index,477);
        W_132 = W_132+ a *Ghimj(index,478);
        W_134 = W_134+ a *Ghimj(index,479);
        W_138 = W_138+ a *Ghimj(index,480);
        a = - W_97/ Ghimj(index,549);
        W_97 = -a;
        W_98 = W_98+ a *Ghimj(index,550);
        W_120 = W_120+ a *Ghimj(index,551);
        W_122 = W_122+ a *Ghimj(index,552);
        W_126 = W_126+ a *Ghimj(index,553);
        W_127 = W_127+ a *Ghimj(index,554);
        W_130 = W_130+ a *Ghimj(index,555);
        W_137 = W_137+ a *Ghimj(index,556);
        a = - W_98/ Ghimj(index,557);
        W_98 = -a;
        W_107 = W_107+ a *Ghimj(index,558);
        W_120 = W_120+ a *Ghimj(index,559);
        W_124 = W_124+ a *Ghimj(index,560);
        W_126 = W_126+ a *Ghimj(index,561);
        W_127 = W_127+ a *Ghimj(index,562);
        a = - W_100/ Ghimj(index,573);
        W_100 = -a;
        W_105 = W_105+ a *Ghimj(index,574);
        W_112 = W_112+ a *Ghimj(index,575);
        W_116 = W_116+ a *Ghimj(index,576);
        W_118 = W_118+ a *Ghimj(index,577);
        W_123 = W_123+ a *Ghimj(index,578);
        W_126 = W_126+ a *Ghimj(index,579);
        W_127 = W_127+ a *Ghimj(index,580);
        W_129 = W_129+ a *Ghimj(index,581);
        W_132 = W_132+ a *Ghimj(index,582);
        W_134 = W_134+ a *Ghimj(index,583);
        W_138 = W_138+ a *Ghimj(index,584);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_112/ Ghimj(index,677);
        W_112 = -a;
        W_116 = W_116+ a *Ghimj(index,678);
        W_123 = W_123+ a *Ghimj(index,679);
        W_126 = W_126+ a *Ghimj(index,680);
        W_128 = W_128+ a *Ghimj(index,681);
        W_134 = W_134+ a *Ghimj(index,682);
        W_137 = W_137+ a *Ghimj(index,683);
        W_138 = W_138+ a *Ghimj(index,684);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_118/ Ghimj(index,745);
        W_118 = -a;
        W_123 = W_123+ a *Ghimj(index,746);
        W_125 = W_125+ a *Ghimj(index,747);
        W_126 = W_126+ a *Ghimj(index,748);
        W_127 = W_127+ a *Ghimj(index,749);
        W_128 = W_128+ a *Ghimj(index,750);
        W_129 = W_129+ a *Ghimj(index,751);
        W_131 = W_131+ a *Ghimj(index,752);
        W_132 = W_132+ a *Ghimj(index,753);
        W_134 = W_134+ a *Ghimj(index,754);
        W_135 = W_135+ a *Ghimj(index,755);
        W_137 = W_137+ a *Ghimj(index,756);
        W_138 = W_138+ a *Ghimj(index,757);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        Ghimj(index,1036) = W_1;
        Ghimj(index,1037) = W_39;
        Ghimj(index,1038) = W_41;
        Ghimj(index,1039) = W_42;
        Ghimj(index,1040) = W_43;
        Ghimj(index,1041) = W_50;
        Ghimj(index,1042) = W_52;
        Ghimj(index,1043) = W_54;
        Ghimj(index,1044) = W_55;
        Ghimj(index,1045) = W_57;
        Ghimj(index,1046) = W_75;
        Ghimj(index,1047) = W_80;
        Ghimj(index,1048) = W_83;
        Ghimj(index,1049) = W_88;
        Ghimj(index,1050) = W_90;
        Ghimj(index,1051) = W_97;
        Ghimj(index,1052) = W_98;
        Ghimj(index,1053) = W_100;
        Ghimj(index,1054) = W_103;
        Ghimj(index,1055) = W_104;
        Ghimj(index,1056) = W_105;
        Ghimj(index,1057) = W_106;
        Ghimj(index,1058) = W_107;
        Ghimj(index,1059) = W_112;
        Ghimj(index,1060) = W_114;
        Ghimj(index,1061) = W_116;
        Ghimj(index,1062) = W_118;
        Ghimj(index,1063) = W_119;
        Ghimj(index,1064) = W_120;
        Ghimj(index,1065) = W_121;
        Ghimj(index,1066) = W_122;
        Ghimj(index,1067) = W_123;
        Ghimj(index,1068) = W_124;
        Ghimj(index,1069) = W_125;
        Ghimj(index,1070) = W_126;
        Ghimj(index,1071) = W_127;
        Ghimj(index,1072) = W_128;
        Ghimj(index,1073) = W_129;
        Ghimj(index,1074) = W_130;
        Ghimj(index,1075) = W_131;
        Ghimj(index,1076) = W_132;
        Ghimj(index,1077) = W_133;
        Ghimj(index,1078) = W_134;
        Ghimj(index,1079) = W_135;
        Ghimj(index,1080) = W_136;
        Ghimj(index,1081) = W_137;
        Ghimj(index,1082) = W_138;
        W_40 = Ghimj(index,1083);
        W_44 = Ghimj(index,1084);
        W_45 = Ghimj(index,1085);
        W_47 = Ghimj(index,1086);
        W_48 = Ghimj(index,1087);
        W_49 = Ghimj(index,1088);
        W_52 = Ghimj(index,1089);
        W_53 = Ghimj(index,1090);
        W_54 = Ghimj(index,1091);
        W_55 = Ghimj(index,1092);
        W_57 = Ghimj(index,1093);
        W_61 = Ghimj(index,1094);
        W_63 = Ghimj(index,1095);
        W_67 = Ghimj(index,1096);
        W_70 = Ghimj(index,1097);
        W_73 = Ghimj(index,1098);
        W_74 = Ghimj(index,1099);
        W_75 = Ghimj(index,1100);
        W_76 = Ghimj(index,1101);
        W_77 = Ghimj(index,1102);
        W_78 = Ghimj(index,1103);
        W_79 = Ghimj(index,1104);
        W_83 = Ghimj(index,1105);
        W_84 = Ghimj(index,1106);
        W_86 = Ghimj(index,1107);
        W_87 = Ghimj(index,1108);
        W_88 = Ghimj(index,1109);
        W_92 = Ghimj(index,1110);
        W_93 = Ghimj(index,1111);
        W_97 = Ghimj(index,1112);
        W_98 = Ghimj(index,1113);
        W_101 = Ghimj(index,1114);
        W_102 = Ghimj(index,1115);
        W_103 = Ghimj(index,1116);
        W_104 = Ghimj(index,1117);
        W_105 = Ghimj(index,1118);
        W_106 = Ghimj(index,1119);
        W_107 = Ghimj(index,1120);
        W_110 = Ghimj(index,1121);
        W_111 = Ghimj(index,1122);
        W_112 = Ghimj(index,1123);
        W_114 = Ghimj(index,1124);
        W_115 = Ghimj(index,1125);
        W_116 = Ghimj(index,1126);
        W_117 = Ghimj(index,1127);
        W_118 = Ghimj(index,1128);
        W_119 = Ghimj(index,1129);
        W_120 = Ghimj(index,1130);
        W_121 = Ghimj(index,1131);
        W_122 = Ghimj(index,1132);
        W_123 = Ghimj(index,1133);
        W_124 = Ghimj(index,1134);
        W_125 = Ghimj(index,1135);
        W_126 = Ghimj(index,1136);
        W_127 = Ghimj(index,1137);
        W_128 = Ghimj(index,1138);
        W_129 = Ghimj(index,1139);
        W_130 = Ghimj(index,1140);
        W_131 = Ghimj(index,1141);
        W_132 = Ghimj(index,1142);
        W_133 = Ghimj(index,1143);
        W_134 = Ghimj(index,1144);
        W_135 = Ghimj(index,1145);
        W_136 = Ghimj(index,1146);
        W_137 = Ghimj(index,1147);
        W_138 = Ghimj(index,1148);
        a = - W_40/ Ghimj(index,260);
        W_40 = -a;
        W_126 = W_126+ a *Ghimj(index,261);
        a = - W_44/ Ghimj(index,268);
        W_44 = -a;
        W_126 = W_126+ a *Ghimj(index,269);
        a = - W_45/ Ghimj(index,270);
        W_45 = -a;
        W_126 = W_126+ a *Ghimj(index,271);
        a = - W_47/ Ghimj(index,276);
        W_47 = -a;
        W_126 = W_126+ a *Ghimj(index,277);
        a = - W_48/ Ghimj(index,278);
        W_48 = -a;
        W_126 = W_126+ a *Ghimj(index,279);
        a = - W_49/ Ghimj(index,280);
        W_49 = -a;
        W_126 = W_126+ a *Ghimj(index,281);
        a = - W_52/ Ghimj(index,288);
        W_52 = -a;
        W_126 = W_126+ a *Ghimj(index,289);
        a = - W_53/ Ghimj(index,290);
        W_53 = -a;
        W_126 = W_126+ a *Ghimj(index,291);
        a = - W_54/ Ghimj(index,292);
        W_54 = -a;
        W_126 = W_126+ a *Ghimj(index,293);
        a = - W_55/ Ghimj(index,294);
        W_55 = -a;
        W_126 = W_126+ a *Ghimj(index,295);
        a = - W_57/ Ghimj(index,300);
        W_57 = -a;
        W_120 = W_120+ a *Ghimj(index,301);
        W_126 = W_126+ a *Ghimj(index,302);
        a = - W_61/ Ghimj(index,315);
        W_61 = -a;
        W_70 = W_70+ a *Ghimj(index,316);
        W_87 = W_87+ a *Ghimj(index,317);
        W_126 = W_126+ a *Ghimj(index,318);
        a = - W_63/ Ghimj(index,323);
        W_63 = -a;
        W_121 = W_121+ a *Ghimj(index,324);
        W_126 = W_126+ a *Ghimj(index,325);
        W_137 = W_137+ a *Ghimj(index,326);
        a = - W_67/ Ghimj(index,339);
        W_67 = -a;
        W_115 = W_115+ a *Ghimj(index,340);
        W_126 = W_126+ a *Ghimj(index,341);
        W_137 = W_137+ a *Ghimj(index,342);
        a = - W_70/ Ghimj(index,352);
        W_70 = -a;
        W_84 = W_84+ a *Ghimj(index,353);
        W_87 = W_87+ a *Ghimj(index,354);
        W_126 = W_126+ a *Ghimj(index,355);
        a = - W_73/ Ghimj(index,364);
        W_73 = -a;
        W_126 = W_126+ a *Ghimj(index,365);
        W_135 = W_135+ a *Ghimj(index,366);
        W_137 = W_137+ a *Ghimj(index,367);
        a = - W_74/ Ghimj(index,368);
        W_74 = -a;
        W_117 = W_117+ a *Ghimj(index,369);
        W_121 = W_121+ a *Ghimj(index,370);
        W_125 = W_125+ a *Ghimj(index,371);
        W_126 = W_126+ a *Ghimj(index,372);
        W_137 = W_137+ a *Ghimj(index,373);
        a = - W_75/ Ghimj(index,374);
        W_75 = -a;
        W_120 = W_120+ a *Ghimj(index,375);
        W_126 = W_126+ a *Ghimj(index,376);
        a = - W_76/ Ghimj(index,377);
        W_76 = -a;
        W_87 = W_87+ a *Ghimj(index,378);
        W_126 = W_126+ a *Ghimj(index,379);
        W_133 = W_133+ a *Ghimj(index,380);
        W_135 = W_135+ a *Ghimj(index,381);
        a = - W_77/ Ghimj(index,382);
        W_77 = -a;
        W_121 = W_121+ a *Ghimj(index,383);
        W_126 = W_126+ a *Ghimj(index,384);
        W_135 = W_135+ a *Ghimj(index,385);
        a = - W_78/ Ghimj(index,386);
        W_78 = -a;
        W_103 = W_103+ a *Ghimj(index,387);
        W_106 = W_106+ a *Ghimj(index,388);
        W_107 = W_107+ a *Ghimj(index,389);
        W_110 = W_110+ a *Ghimj(index,390);
        W_124 = W_124+ a *Ghimj(index,391);
        W_126 = W_126+ a *Ghimj(index,392);
        a = - W_79/ Ghimj(index,393);
        W_79 = -a;
        W_102 = W_102+ a *Ghimj(index,394);
        W_126 = W_126+ a *Ghimj(index,395);
        W_137 = W_137+ a *Ghimj(index,396);
        a = - W_83/ Ghimj(index,416);
        W_83 = -a;
        W_128 = W_128+ a *Ghimj(index,417);
        W_135 = W_135+ a *Ghimj(index,418);
        W_136 = W_136+ a *Ghimj(index,419);
        W_138 = W_138+ a *Ghimj(index,420);
        a = - W_84/ Ghimj(index,421);
        W_84 = -a;
        W_92 = W_92+ a *Ghimj(index,422);
        W_124 = W_124+ a *Ghimj(index,423);
        W_135 = W_135+ a *Ghimj(index,424);
        W_137 = W_137+ a *Ghimj(index,425);
        a = - W_86/ Ghimj(index,436);
        W_86 = -a;
        W_93 = W_93+ a *Ghimj(index,437);
        W_125 = W_125+ a *Ghimj(index,438);
        W_126 = W_126+ a *Ghimj(index,439);
        W_133 = W_133+ a *Ghimj(index,440);
        W_137 = W_137+ a *Ghimj(index,441);
        a = - W_87/ Ghimj(index,444);
        W_87 = -a;
        W_92 = W_92+ a *Ghimj(index,445);
        W_124 = W_124+ a *Ghimj(index,446);
        W_126 = W_126+ a *Ghimj(index,447);
        W_135 = W_135+ a *Ghimj(index,448);
        W_137 = W_137+ a *Ghimj(index,449);
        a = - W_88/ Ghimj(index,450);
        W_88 = -a;
        W_103 = W_103+ a *Ghimj(index,451);
        W_106 = W_106+ a *Ghimj(index,452);
        W_124 = W_124+ a *Ghimj(index,453);
        W_126 = W_126+ a *Ghimj(index,454);
        W_127 = W_127+ a *Ghimj(index,455);
        W_137 = W_137+ a *Ghimj(index,456);
        a = - W_92/ Ghimj(index,489);
        W_92 = -a;
        W_124 = W_124+ a *Ghimj(index,490);
        W_126 = W_126+ a *Ghimj(index,491);
        W_133 = W_133+ a *Ghimj(index,492);
        W_135 = W_135+ a *Ghimj(index,493);
        W_137 = W_137+ a *Ghimj(index,494);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        a = - W_97/ Ghimj(index,549);
        W_97 = -a;
        W_98 = W_98+ a *Ghimj(index,550);
        W_120 = W_120+ a *Ghimj(index,551);
        W_122 = W_122+ a *Ghimj(index,552);
        W_126 = W_126+ a *Ghimj(index,553);
        W_127 = W_127+ a *Ghimj(index,554);
        W_130 = W_130+ a *Ghimj(index,555);
        W_137 = W_137+ a *Ghimj(index,556);
        a = - W_98/ Ghimj(index,557);
        W_98 = -a;
        W_107 = W_107+ a *Ghimj(index,558);
        W_120 = W_120+ a *Ghimj(index,559);
        W_124 = W_124+ a *Ghimj(index,560);
        W_126 = W_126+ a *Ghimj(index,561);
        W_127 = W_127+ a *Ghimj(index,562);
        a = - W_101/ Ghimj(index,586);
        W_101 = -a;
        W_105 = W_105+ a *Ghimj(index,587);
        W_114 = W_114+ a *Ghimj(index,588);
        W_116 = W_116+ a *Ghimj(index,589);
        W_119 = W_119+ a *Ghimj(index,590);
        W_123 = W_123+ a *Ghimj(index,591);
        W_126 = W_126+ a *Ghimj(index,592);
        W_128 = W_128+ a *Ghimj(index,593);
        W_130 = W_130+ a *Ghimj(index,594);
        W_135 = W_135+ a *Ghimj(index,595);
        W_136 = W_136+ a *Ghimj(index,596);
        W_138 = W_138+ a *Ghimj(index,597);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_110/ Ghimj(index,659);
        W_110 = -a;
        W_124 = W_124+ a *Ghimj(index,660);
        W_125 = W_125+ a *Ghimj(index,661);
        W_126 = W_126+ a *Ghimj(index,662);
        W_133 = W_133+ a *Ghimj(index,663);
        W_136 = W_136+ a *Ghimj(index,664);
        W_137 = W_137+ a *Ghimj(index,665);
        a = - W_111/ Ghimj(index,669);
        W_111 = -a;
        W_115 = W_115+ a *Ghimj(index,670);
        W_124 = W_124+ a *Ghimj(index,671);
        W_125 = W_125+ a *Ghimj(index,672);
        W_126 = W_126+ a *Ghimj(index,673);
        W_133 = W_133+ a *Ghimj(index,674);
        W_136 = W_136+ a *Ghimj(index,675);
        W_137 = W_137+ a *Ghimj(index,676);
        a = - W_112/ Ghimj(index,677);
        W_112 = -a;
        W_116 = W_116+ a *Ghimj(index,678);
        W_123 = W_123+ a *Ghimj(index,679);
        W_126 = W_126+ a *Ghimj(index,680);
        W_128 = W_128+ a *Ghimj(index,681);
        W_134 = W_134+ a *Ghimj(index,682);
        W_137 = W_137+ a *Ghimj(index,683);
        W_138 = W_138+ a *Ghimj(index,684);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_118/ Ghimj(index,745);
        W_118 = -a;
        W_123 = W_123+ a *Ghimj(index,746);
        W_125 = W_125+ a *Ghimj(index,747);
        W_126 = W_126+ a *Ghimj(index,748);
        W_127 = W_127+ a *Ghimj(index,749);
        W_128 = W_128+ a *Ghimj(index,750);
        W_129 = W_129+ a *Ghimj(index,751);
        W_131 = W_131+ a *Ghimj(index,752);
        W_132 = W_132+ a *Ghimj(index,753);
        W_134 = W_134+ a *Ghimj(index,754);
        W_135 = W_135+ a *Ghimj(index,755);
        W_137 = W_137+ a *Ghimj(index,756);
        W_138 = W_138+ a *Ghimj(index,757);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        Ghimj(index,1083) = W_40;
        Ghimj(index,1084) = W_44;
        Ghimj(index,1085) = W_45;
        Ghimj(index,1086) = W_47;
        Ghimj(index,1087) = W_48;
        Ghimj(index,1088) = W_49;
        Ghimj(index,1089) = W_52;
        Ghimj(index,1090) = W_53;
        Ghimj(index,1091) = W_54;
        Ghimj(index,1092) = W_55;
        Ghimj(index,1093) = W_57;
        Ghimj(index,1094) = W_61;
        Ghimj(index,1095) = W_63;
        Ghimj(index,1096) = W_67;
        Ghimj(index,1097) = W_70;
        Ghimj(index,1098) = W_73;
        Ghimj(index,1099) = W_74;
        Ghimj(index,1100) = W_75;
        Ghimj(index,1101) = W_76;
        Ghimj(index,1102) = W_77;
        Ghimj(index,1103) = W_78;
        Ghimj(index,1104) = W_79;
        Ghimj(index,1105) = W_83;
        Ghimj(index,1106) = W_84;
        Ghimj(index,1107) = W_86;
        Ghimj(index,1108) = W_87;
        Ghimj(index,1109) = W_88;
        Ghimj(index,1110) = W_92;
        Ghimj(index,1111) = W_93;
        Ghimj(index,1112) = W_97;
        Ghimj(index,1113) = W_98;
        Ghimj(index,1114) = W_101;
        Ghimj(index,1115) = W_102;
        Ghimj(index,1116) = W_103;
        Ghimj(index,1117) = W_104;
        Ghimj(index,1118) = W_105;
        Ghimj(index,1119) = W_106;
        Ghimj(index,1120) = W_107;
        Ghimj(index,1121) = W_110;
        Ghimj(index,1122) = W_111;
        Ghimj(index,1123) = W_112;
        Ghimj(index,1124) = W_114;
        Ghimj(index,1125) = W_115;
        Ghimj(index,1126) = W_116;
        Ghimj(index,1127) = W_117;
        Ghimj(index,1128) = W_118;
        Ghimj(index,1129) = W_119;
        Ghimj(index,1130) = W_120;
        Ghimj(index,1131) = W_121;
        Ghimj(index,1132) = W_122;
        Ghimj(index,1133) = W_123;
        Ghimj(index,1134) = W_124;
        Ghimj(index,1135) = W_125;
        Ghimj(index,1136) = W_126;
        Ghimj(index,1137) = W_127;
        Ghimj(index,1138) = W_128;
        Ghimj(index,1139) = W_129;
        Ghimj(index,1140) = W_130;
        Ghimj(index,1141) = W_131;
        Ghimj(index,1142) = W_132;
        Ghimj(index,1143) = W_133;
        Ghimj(index,1144) = W_134;
        Ghimj(index,1145) = W_135;
        Ghimj(index,1146) = W_136;
        Ghimj(index,1147) = W_137;
        Ghimj(index,1148) = W_138;
        W_0 = Ghimj(index,1149);
        W_1 = Ghimj(index,1150);
        W_2 = Ghimj(index,1151);
        W_44 = Ghimj(index,1152);
        W_45 = Ghimj(index,1153);
        W_52 = Ghimj(index,1154);
        W_53 = Ghimj(index,1155);
        W_54 = Ghimj(index,1156);
        W_55 = Ghimj(index,1157);
        W_80 = Ghimj(index,1158);
        W_90 = Ghimj(index,1159);
        W_100 = Ghimj(index,1160);
        W_103 = Ghimj(index,1161);
        W_104 = Ghimj(index,1162);
        W_105 = Ghimj(index,1163);
        W_112 = Ghimj(index,1164);
        W_114 = Ghimj(index,1165);
        W_116 = Ghimj(index,1166);
        W_118 = Ghimj(index,1167);
        W_119 = Ghimj(index,1168);
        W_121 = Ghimj(index,1169);
        W_123 = Ghimj(index,1170);
        W_124 = Ghimj(index,1171);
        W_125 = Ghimj(index,1172);
        W_126 = Ghimj(index,1173);
        W_127 = Ghimj(index,1174);
        W_128 = Ghimj(index,1175);
        W_129 = Ghimj(index,1176);
        W_130 = Ghimj(index,1177);
        W_131 = Ghimj(index,1178);
        W_132 = Ghimj(index,1179);
        W_133 = Ghimj(index,1180);
        W_134 = Ghimj(index,1181);
        W_135 = Ghimj(index,1182);
        W_136 = Ghimj(index,1183);
        W_137 = Ghimj(index,1184);
        W_138 = Ghimj(index,1185);
        a = - W_0/ Ghimj(index,0);
        W_0 = -a;
        a = - W_1/ Ghimj(index,1);
        W_1 = -a;
        a = - W_2/ Ghimj(index,2);
        W_2 = -a;
        a = - W_44/ Ghimj(index,268);
        W_44 = -a;
        W_126 = W_126+ a *Ghimj(index,269);
        a = - W_45/ Ghimj(index,270);
        W_45 = -a;
        W_126 = W_126+ a *Ghimj(index,271);
        a = - W_52/ Ghimj(index,288);
        W_52 = -a;
        W_126 = W_126+ a *Ghimj(index,289);
        a = - W_53/ Ghimj(index,290);
        W_53 = -a;
        W_126 = W_126+ a *Ghimj(index,291);
        a = - W_54/ Ghimj(index,292);
        W_54 = -a;
        W_126 = W_126+ a *Ghimj(index,293);
        a = - W_55/ Ghimj(index,294);
        W_55 = -a;
        W_126 = W_126+ a *Ghimj(index,295);
        a = - W_80/ Ghimj(index,397);
        W_80 = -a;
        W_90 = W_90+ a *Ghimj(index,398);
        W_112 = W_112+ a *Ghimj(index,399);
        W_116 = W_116+ a *Ghimj(index,400);
        W_127 = W_127+ a *Ghimj(index,401);
        W_129 = W_129+ a *Ghimj(index,402);
        W_134 = W_134+ a *Ghimj(index,403);
        W_138 = W_138+ a *Ghimj(index,404);
        a = - W_90/ Ghimj(index,469);
        W_90 = -a;
        W_100 = W_100+ a *Ghimj(index,470);
        W_105 = W_105+ a *Ghimj(index,471);
        W_112 = W_112+ a *Ghimj(index,472);
        W_116 = W_116+ a *Ghimj(index,473);
        W_118 = W_118+ a *Ghimj(index,474);
        W_123 = W_123+ a *Ghimj(index,475);
        W_127 = W_127+ a *Ghimj(index,476);
        W_129 = W_129+ a *Ghimj(index,477);
        W_132 = W_132+ a *Ghimj(index,478);
        W_134 = W_134+ a *Ghimj(index,479);
        W_138 = W_138+ a *Ghimj(index,480);
        a = - W_100/ Ghimj(index,573);
        W_100 = -a;
        W_105 = W_105+ a *Ghimj(index,574);
        W_112 = W_112+ a *Ghimj(index,575);
        W_116 = W_116+ a *Ghimj(index,576);
        W_118 = W_118+ a *Ghimj(index,577);
        W_123 = W_123+ a *Ghimj(index,578);
        W_126 = W_126+ a *Ghimj(index,579);
        W_127 = W_127+ a *Ghimj(index,580);
        W_129 = W_129+ a *Ghimj(index,581);
        W_132 = W_132+ a *Ghimj(index,582);
        W_134 = W_134+ a *Ghimj(index,583);
        W_138 = W_138+ a *Ghimj(index,584);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_112/ Ghimj(index,677);
        W_112 = -a;
        W_116 = W_116+ a *Ghimj(index,678);
        W_123 = W_123+ a *Ghimj(index,679);
        W_126 = W_126+ a *Ghimj(index,680);
        W_128 = W_128+ a *Ghimj(index,681);
        W_134 = W_134+ a *Ghimj(index,682);
        W_137 = W_137+ a *Ghimj(index,683);
        W_138 = W_138+ a *Ghimj(index,684);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_118/ Ghimj(index,745);
        W_118 = -a;
        W_123 = W_123+ a *Ghimj(index,746);
        W_125 = W_125+ a *Ghimj(index,747);
        W_126 = W_126+ a *Ghimj(index,748);
        W_127 = W_127+ a *Ghimj(index,749);
        W_128 = W_128+ a *Ghimj(index,750);
        W_129 = W_129+ a *Ghimj(index,751);
        W_131 = W_131+ a *Ghimj(index,752);
        W_132 = W_132+ a *Ghimj(index,753);
        W_134 = W_134+ a *Ghimj(index,754);
        W_135 = W_135+ a *Ghimj(index,755);
        W_137 = W_137+ a *Ghimj(index,756);
        W_138 = W_138+ a *Ghimj(index,757);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        Ghimj(index,1149) = W_0;
        Ghimj(index,1150) = W_1;
        Ghimj(index,1151) = W_2;
        Ghimj(index,1152) = W_44;
        Ghimj(index,1153) = W_45;
        Ghimj(index,1154) = W_52;
        Ghimj(index,1155) = W_53;
        Ghimj(index,1156) = W_54;
        Ghimj(index,1157) = W_55;
        Ghimj(index,1158) = W_80;
        Ghimj(index,1159) = W_90;
        Ghimj(index,1160) = W_100;
        Ghimj(index,1161) = W_103;
        Ghimj(index,1162) = W_104;
        Ghimj(index,1163) = W_105;
        Ghimj(index,1164) = W_112;
        Ghimj(index,1165) = W_114;
        Ghimj(index,1166) = W_116;
        Ghimj(index,1167) = W_118;
        Ghimj(index,1168) = W_119;
        Ghimj(index,1169) = W_121;
        Ghimj(index,1170) = W_123;
        Ghimj(index,1171) = W_124;
        Ghimj(index,1172) = W_125;
        Ghimj(index,1173) = W_126;
        Ghimj(index,1174) = W_127;
        Ghimj(index,1175) = W_128;
        Ghimj(index,1176) = W_129;
        Ghimj(index,1177) = W_130;
        Ghimj(index,1178) = W_131;
        Ghimj(index,1179) = W_132;
        Ghimj(index,1180) = W_133;
        Ghimj(index,1181) = W_134;
        Ghimj(index,1182) = W_135;
        Ghimj(index,1183) = W_136;
        Ghimj(index,1184) = W_137;
        Ghimj(index,1185) = W_138;
        W_58 = Ghimj(index,1186);
        W_65 = Ghimj(index,1187);
        W_66 = Ghimj(index,1188);
        W_72 = Ghimj(index,1189);
        W_77 = Ghimj(index,1190);
        W_82 = Ghimj(index,1191);
        W_89 = Ghimj(index,1192);
        W_91 = Ghimj(index,1193);
        W_93 = Ghimj(index,1194);
        W_94 = Ghimj(index,1195);
        W_98 = Ghimj(index,1196);
        W_102 = Ghimj(index,1197);
        W_103 = Ghimj(index,1198);
        W_104 = Ghimj(index,1199);
        W_106 = Ghimj(index,1200);
        W_107 = Ghimj(index,1201);
        W_108 = Ghimj(index,1202);
        W_109 = Ghimj(index,1203);
        W_110 = Ghimj(index,1204);
        W_113 = Ghimj(index,1205);
        W_114 = Ghimj(index,1206);
        W_115 = Ghimj(index,1207);
        W_117 = Ghimj(index,1208);
        W_120 = Ghimj(index,1209);
        W_121 = Ghimj(index,1210);
        W_122 = Ghimj(index,1211);
        W_124 = Ghimj(index,1212);
        W_125 = Ghimj(index,1213);
        W_126 = Ghimj(index,1214);
        W_127 = Ghimj(index,1215);
        W_128 = Ghimj(index,1216);
        W_129 = Ghimj(index,1217);
        W_130 = Ghimj(index,1218);
        W_131 = Ghimj(index,1219);
        W_132 = Ghimj(index,1220);
        W_133 = Ghimj(index,1221);
        W_134 = Ghimj(index,1222);
        W_135 = Ghimj(index,1223);
        W_136 = Ghimj(index,1224);
        W_137 = Ghimj(index,1225);
        W_138 = Ghimj(index,1226);
        a = - W_58/ Ghimj(index,303);
        W_58 = -a;
        W_91 = W_91+ a *Ghimj(index,304);
        W_126 = W_126+ a *Ghimj(index,305);
        a = - W_65/ Ghimj(index,331);
        W_65 = -a;
        W_114 = W_114+ a *Ghimj(index,332);
        W_126 = W_126+ a *Ghimj(index,333);
        W_132 = W_132+ a *Ghimj(index,334);
        a = - W_66/ Ghimj(index,335);
        W_66 = -a;
        W_109 = W_109+ a *Ghimj(index,336);
        W_126 = W_126+ a *Ghimj(index,337);
        W_137 = W_137+ a *Ghimj(index,338);
        a = - W_72/ Ghimj(index,360);
        W_72 = -a;
        W_94 = W_94+ a *Ghimj(index,361);
        W_126 = W_126+ a *Ghimj(index,362);
        W_137 = W_137+ a *Ghimj(index,363);
        a = - W_77/ Ghimj(index,382);
        W_77 = -a;
        W_121 = W_121+ a *Ghimj(index,383);
        W_126 = W_126+ a *Ghimj(index,384);
        W_135 = W_135+ a *Ghimj(index,385);
        a = - W_82/ Ghimj(index,412);
        W_82 = -a;
        W_113 = W_113+ a *Ghimj(index,413);
        W_126 = W_126+ a *Ghimj(index,414);
        W_137 = W_137+ a *Ghimj(index,415);
        a = - W_89/ Ghimj(index,457);
        W_89 = -a;
        W_93 = W_93+ a *Ghimj(index,458);
        W_94 = W_94+ a *Ghimj(index,459);
        W_102 = W_102+ a *Ghimj(index,460);
        W_107 = W_107+ a *Ghimj(index,461);
        W_109 = W_109+ a *Ghimj(index,462);
        W_113 = W_113+ a *Ghimj(index,463);
        W_117 = W_117+ a *Ghimj(index,464);
        W_124 = W_124+ a *Ghimj(index,465);
        W_125 = W_125+ a *Ghimj(index,466);
        W_126 = W_126+ a *Ghimj(index,467);
        a = - W_91/ Ghimj(index,481);
        W_91 = -a;
        W_106 = W_106+ a *Ghimj(index,482);
        W_109 = W_109+ a *Ghimj(index,483);
        W_126 = W_126+ a *Ghimj(index,484);
        W_133 = W_133+ a *Ghimj(index,485);
        W_136 = W_136+ a *Ghimj(index,486);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        a = - W_98/ Ghimj(index,557);
        W_98 = -a;
        W_107 = W_107+ a *Ghimj(index,558);
        W_120 = W_120+ a *Ghimj(index,559);
        W_124 = W_124+ a *Ghimj(index,560);
        W_126 = W_126+ a *Ghimj(index,561);
        W_127 = W_127+ a *Ghimj(index,562);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_108/ Ghimj(index,636);
        W_108 = -a;
        W_109 = W_109+ a *Ghimj(index,637);
        W_113 = W_113+ a *Ghimj(index,638);
        W_115 = W_115+ a *Ghimj(index,639);
        W_124 = W_124+ a *Ghimj(index,640);
        W_125 = W_125+ a *Ghimj(index,641);
        W_126 = W_126+ a *Ghimj(index,642);
        W_133 = W_133+ a *Ghimj(index,643);
        W_135 = W_135+ a *Ghimj(index,644);
        W_136 = W_136+ a *Ghimj(index,645);
        W_137 = W_137+ a *Ghimj(index,646);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        a = - W_110/ Ghimj(index,659);
        W_110 = -a;
        W_124 = W_124+ a *Ghimj(index,660);
        W_125 = W_125+ a *Ghimj(index,661);
        W_126 = W_126+ a *Ghimj(index,662);
        W_133 = W_133+ a *Ghimj(index,663);
        W_136 = W_136+ a *Ghimj(index,664);
        W_137 = W_137+ a *Ghimj(index,665);
        a = - W_113/ Ghimj(index,689);
        W_113 = -a;
        W_124 = W_124+ a *Ghimj(index,690);
        W_125 = W_125+ a *Ghimj(index,691);
        W_126 = W_126+ a *Ghimj(index,692);
        W_133 = W_133+ a *Ghimj(index,693);
        W_135 = W_135+ a *Ghimj(index,694);
        W_136 = W_136+ a *Ghimj(index,695);
        W_137 = W_137+ a *Ghimj(index,696);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        Ghimj(index,1186) = W_58;
        Ghimj(index,1187) = W_65;
        Ghimj(index,1188) = W_66;
        Ghimj(index,1189) = W_72;
        Ghimj(index,1190) = W_77;
        Ghimj(index,1191) = W_82;
        Ghimj(index,1192) = W_89;
        Ghimj(index,1193) = W_91;
        Ghimj(index,1194) = W_93;
        Ghimj(index,1195) = W_94;
        Ghimj(index,1196) = W_98;
        Ghimj(index,1197) = W_102;
        Ghimj(index,1198) = W_103;
        Ghimj(index,1199) = W_104;
        Ghimj(index,1200) = W_106;
        Ghimj(index,1201) = W_107;
        Ghimj(index,1202) = W_108;
        Ghimj(index,1203) = W_109;
        Ghimj(index,1204) = W_110;
        Ghimj(index,1205) = W_113;
        Ghimj(index,1206) = W_114;
        Ghimj(index,1207) = W_115;
        Ghimj(index,1208) = W_117;
        Ghimj(index,1209) = W_120;
        Ghimj(index,1210) = W_121;
        Ghimj(index,1211) = W_122;
        Ghimj(index,1212) = W_124;
        Ghimj(index,1213) = W_125;
        Ghimj(index,1214) = W_126;
        Ghimj(index,1215) = W_127;
        Ghimj(index,1216) = W_128;
        Ghimj(index,1217) = W_129;
        Ghimj(index,1218) = W_130;
        Ghimj(index,1219) = W_131;
        Ghimj(index,1220) = W_132;
        Ghimj(index,1221) = W_133;
        Ghimj(index,1222) = W_134;
        Ghimj(index,1223) = W_135;
        Ghimj(index,1224) = W_136;
        Ghimj(index,1225) = W_137;
        Ghimj(index,1226) = W_138;
        W_51 = Ghimj(index,1227);
        W_59 = Ghimj(index,1228);
        W_75 = Ghimj(index,1229);
        W_116 = Ghimj(index,1230);
        W_118 = Ghimj(index,1231);
        W_120 = Ghimj(index,1232);
        W_122 = Ghimj(index,1233);
        W_123 = Ghimj(index,1234);
        W_124 = Ghimj(index,1235);
        W_125 = Ghimj(index,1236);
        W_126 = Ghimj(index,1237);
        W_127 = Ghimj(index,1238);
        W_128 = Ghimj(index,1239);
        W_129 = Ghimj(index,1240);
        W_130 = Ghimj(index,1241);
        W_131 = Ghimj(index,1242);
        W_132 = Ghimj(index,1243);
        W_133 = Ghimj(index,1244);
        W_134 = Ghimj(index,1245);
        W_135 = Ghimj(index,1246);
        W_136 = Ghimj(index,1247);
        W_137 = Ghimj(index,1248);
        W_138 = Ghimj(index,1249);
        a = - W_51/ Ghimj(index,285);
        W_51 = -a;
        W_132 = W_132+ a *Ghimj(index,286);
        W_134 = W_134+ a *Ghimj(index,287);
        a = - W_59/ Ghimj(index,306);
        W_59 = -a;
        W_133 = W_133+ a *Ghimj(index,307);
        W_135 = W_135+ a *Ghimj(index,308);
        a = - W_75/ Ghimj(index,374);
        W_75 = -a;
        W_120 = W_120+ a *Ghimj(index,375);
        W_126 = W_126+ a *Ghimj(index,376);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_118/ Ghimj(index,745);
        W_118 = -a;
        W_123 = W_123+ a *Ghimj(index,746);
        W_125 = W_125+ a *Ghimj(index,747);
        W_126 = W_126+ a *Ghimj(index,748);
        W_127 = W_127+ a *Ghimj(index,749);
        W_128 = W_128+ a *Ghimj(index,750);
        W_129 = W_129+ a *Ghimj(index,751);
        W_131 = W_131+ a *Ghimj(index,752);
        W_132 = W_132+ a *Ghimj(index,753);
        W_134 = W_134+ a *Ghimj(index,754);
        W_135 = W_135+ a *Ghimj(index,755);
        W_137 = W_137+ a *Ghimj(index,756);
        W_138 = W_138+ a *Ghimj(index,757);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        a = - W_130/ Ghimj(index,1218);
        W_130 = -a;
        W_131 = W_131+ a *Ghimj(index,1219);
        W_132 = W_132+ a *Ghimj(index,1220);
        W_133 = W_133+ a *Ghimj(index,1221);
        W_134 = W_134+ a *Ghimj(index,1222);
        W_135 = W_135+ a *Ghimj(index,1223);
        W_136 = W_136+ a *Ghimj(index,1224);
        W_137 = W_137+ a *Ghimj(index,1225);
        W_138 = W_138+ a *Ghimj(index,1226);
        Ghimj(index,1227) = W_51;
        Ghimj(index,1228) = W_59;
        Ghimj(index,1229) = W_75;
        Ghimj(index,1230) = W_116;
        Ghimj(index,1231) = W_118;
        Ghimj(index,1232) = W_120;
        Ghimj(index,1233) = W_122;
        Ghimj(index,1234) = W_123;
        Ghimj(index,1235) = W_124;
        Ghimj(index,1236) = W_125;
        Ghimj(index,1237) = W_126;
        Ghimj(index,1238) = W_127;
        Ghimj(index,1239) = W_128;
        Ghimj(index,1240) = W_129;
        Ghimj(index,1241) = W_130;
        Ghimj(index,1242) = W_131;
        Ghimj(index,1243) = W_132;
        Ghimj(index,1244) = W_133;
        Ghimj(index,1245) = W_134;
        Ghimj(index,1246) = W_135;
        Ghimj(index,1247) = W_136;
        Ghimj(index,1248) = W_137;
        Ghimj(index,1249) = W_138;
        W_105 = Ghimj(index,1250);
        W_114 = Ghimj(index,1251);
        W_118 = Ghimj(index,1252);
        W_123 = Ghimj(index,1253);
        W_124 = Ghimj(index,1254);
        W_125 = Ghimj(index,1255);
        W_126 = Ghimj(index,1256);
        W_127 = Ghimj(index,1257);
        W_128 = Ghimj(index,1258);
        W_129 = Ghimj(index,1259);
        W_130 = Ghimj(index,1260);
        W_131 = Ghimj(index,1261);
        W_132 = Ghimj(index,1262);
        W_133 = Ghimj(index,1263);
        W_134 = Ghimj(index,1264);
        W_135 = Ghimj(index,1265);
        W_136 = Ghimj(index,1266);
        W_137 = Ghimj(index,1267);
        W_138 = Ghimj(index,1268);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_118/ Ghimj(index,745);
        W_118 = -a;
        W_123 = W_123+ a *Ghimj(index,746);
        W_125 = W_125+ a *Ghimj(index,747);
        W_126 = W_126+ a *Ghimj(index,748);
        W_127 = W_127+ a *Ghimj(index,749);
        W_128 = W_128+ a *Ghimj(index,750);
        W_129 = W_129+ a *Ghimj(index,751);
        W_131 = W_131+ a *Ghimj(index,752);
        W_132 = W_132+ a *Ghimj(index,753);
        W_134 = W_134+ a *Ghimj(index,754);
        W_135 = W_135+ a *Ghimj(index,755);
        W_137 = W_137+ a *Ghimj(index,756);
        W_138 = W_138+ a *Ghimj(index,757);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        a = - W_130/ Ghimj(index,1218);
        W_130 = -a;
        W_131 = W_131+ a *Ghimj(index,1219);
        W_132 = W_132+ a *Ghimj(index,1220);
        W_133 = W_133+ a *Ghimj(index,1221);
        W_134 = W_134+ a *Ghimj(index,1222);
        W_135 = W_135+ a *Ghimj(index,1223);
        W_136 = W_136+ a *Ghimj(index,1224);
        W_137 = W_137+ a *Ghimj(index,1225);
        W_138 = W_138+ a *Ghimj(index,1226);
        a = - W_131/ Ghimj(index,1242);
        W_131 = -a;
        W_132 = W_132+ a *Ghimj(index,1243);
        W_133 = W_133+ a *Ghimj(index,1244);
        W_134 = W_134+ a *Ghimj(index,1245);
        W_135 = W_135+ a *Ghimj(index,1246);
        W_136 = W_136+ a *Ghimj(index,1247);
        W_137 = W_137+ a *Ghimj(index,1248);
        W_138 = W_138+ a *Ghimj(index,1249);
        Ghimj(index,1250) = W_105;
        Ghimj(index,1251) = W_114;
        Ghimj(index,1252) = W_118;
        Ghimj(index,1253) = W_123;
        Ghimj(index,1254) = W_124;
        Ghimj(index,1255) = W_125;
        Ghimj(index,1256) = W_126;
        Ghimj(index,1257) = W_127;
        Ghimj(index,1258) = W_128;
        Ghimj(index,1259) = W_129;
        Ghimj(index,1260) = W_130;
        Ghimj(index,1261) = W_131;
        Ghimj(index,1262) = W_132;
        Ghimj(index,1263) = W_133;
        Ghimj(index,1264) = W_134;
        Ghimj(index,1265) = W_135;
        Ghimj(index,1266) = W_136;
        Ghimj(index,1267) = W_137;
        Ghimj(index,1268) = W_138;
        W_59 = Ghimj(index,1269);
        W_60 = Ghimj(index,1270);
        W_70 = Ghimj(index,1271);
        W_76 = Ghimj(index,1272);
        W_84 = Ghimj(index,1273);
        W_87 = Ghimj(index,1274);
        W_92 = Ghimj(index,1275);
        W_93 = Ghimj(index,1276);
        W_94 = Ghimj(index,1277);
        W_99 = Ghimj(index,1278);
        W_102 = Ghimj(index,1279);
        W_109 = Ghimj(index,1280);
        W_111 = Ghimj(index,1281);
        W_113 = Ghimj(index,1282);
        W_115 = Ghimj(index,1283);
        W_117 = Ghimj(index,1284);
        W_120 = Ghimj(index,1285);
        W_121 = Ghimj(index,1286);
        W_122 = Ghimj(index,1287);
        W_124 = Ghimj(index,1288);
        W_125 = Ghimj(index,1289);
        W_126 = Ghimj(index,1290);
        W_127 = Ghimj(index,1291);
        W_128 = Ghimj(index,1292);
        W_129 = Ghimj(index,1293);
        W_130 = Ghimj(index,1294);
        W_131 = Ghimj(index,1295);
        W_132 = Ghimj(index,1296);
        W_133 = Ghimj(index,1297);
        W_134 = Ghimj(index,1298);
        W_135 = Ghimj(index,1299);
        W_136 = Ghimj(index,1300);
        W_137 = Ghimj(index,1301);
        W_138 = Ghimj(index,1302);
        a = - W_59/ Ghimj(index,306);
        W_59 = -a;
        W_133 = W_133+ a *Ghimj(index,307);
        W_135 = W_135+ a *Ghimj(index,308);
        a = - W_60/ Ghimj(index,310);
        W_60 = -a;
        W_92 = W_92+ a *Ghimj(index,311);
        W_120 = W_120+ a *Ghimj(index,312);
        W_133 = W_133+ a *Ghimj(index,313);
        W_135 = W_135+ a *Ghimj(index,314);
        a = - W_70/ Ghimj(index,352);
        W_70 = -a;
        W_84 = W_84+ a *Ghimj(index,353);
        W_87 = W_87+ a *Ghimj(index,354);
        W_126 = W_126+ a *Ghimj(index,355);
        a = - W_76/ Ghimj(index,377);
        W_76 = -a;
        W_87 = W_87+ a *Ghimj(index,378);
        W_126 = W_126+ a *Ghimj(index,379);
        W_133 = W_133+ a *Ghimj(index,380);
        W_135 = W_135+ a *Ghimj(index,381);
        a = - W_84/ Ghimj(index,421);
        W_84 = -a;
        W_92 = W_92+ a *Ghimj(index,422);
        W_124 = W_124+ a *Ghimj(index,423);
        W_135 = W_135+ a *Ghimj(index,424);
        W_137 = W_137+ a *Ghimj(index,425);
        a = - W_87/ Ghimj(index,444);
        W_87 = -a;
        W_92 = W_92+ a *Ghimj(index,445);
        W_124 = W_124+ a *Ghimj(index,446);
        W_126 = W_126+ a *Ghimj(index,447);
        W_135 = W_135+ a *Ghimj(index,448);
        W_137 = W_137+ a *Ghimj(index,449);
        a = - W_92/ Ghimj(index,489);
        W_92 = -a;
        W_124 = W_124+ a *Ghimj(index,490);
        W_126 = W_126+ a *Ghimj(index,491);
        W_133 = W_133+ a *Ghimj(index,492);
        W_135 = W_135+ a *Ghimj(index,493);
        W_137 = W_137+ a *Ghimj(index,494);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        a = - W_99/ Ghimj(index,565);
        W_99 = -a;
        W_102 = W_102+ a *Ghimj(index,566);
        W_111 = W_111+ a *Ghimj(index,567);
        W_125 = W_125+ a *Ghimj(index,568);
        W_126 = W_126+ a *Ghimj(index,569);
        W_133 = W_133+ a *Ghimj(index,570);
        W_137 = W_137+ a *Ghimj(index,571);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        a = - W_111/ Ghimj(index,669);
        W_111 = -a;
        W_115 = W_115+ a *Ghimj(index,670);
        W_124 = W_124+ a *Ghimj(index,671);
        W_125 = W_125+ a *Ghimj(index,672);
        W_126 = W_126+ a *Ghimj(index,673);
        W_133 = W_133+ a *Ghimj(index,674);
        W_136 = W_136+ a *Ghimj(index,675);
        W_137 = W_137+ a *Ghimj(index,676);
        a = - W_113/ Ghimj(index,689);
        W_113 = -a;
        W_124 = W_124+ a *Ghimj(index,690);
        W_125 = W_125+ a *Ghimj(index,691);
        W_126 = W_126+ a *Ghimj(index,692);
        W_133 = W_133+ a *Ghimj(index,693);
        W_135 = W_135+ a *Ghimj(index,694);
        W_136 = W_136+ a *Ghimj(index,695);
        W_137 = W_137+ a *Ghimj(index,696);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        a = - W_130/ Ghimj(index,1218);
        W_130 = -a;
        W_131 = W_131+ a *Ghimj(index,1219);
        W_132 = W_132+ a *Ghimj(index,1220);
        W_133 = W_133+ a *Ghimj(index,1221);
        W_134 = W_134+ a *Ghimj(index,1222);
        W_135 = W_135+ a *Ghimj(index,1223);
        W_136 = W_136+ a *Ghimj(index,1224);
        W_137 = W_137+ a *Ghimj(index,1225);
        W_138 = W_138+ a *Ghimj(index,1226);
        a = - W_131/ Ghimj(index,1242);
        W_131 = -a;
        W_132 = W_132+ a *Ghimj(index,1243);
        W_133 = W_133+ a *Ghimj(index,1244);
        W_134 = W_134+ a *Ghimj(index,1245);
        W_135 = W_135+ a *Ghimj(index,1246);
        W_136 = W_136+ a *Ghimj(index,1247);
        W_137 = W_137+ a *Ghimj(index,1248);
        W_138 = W_138+ a *Ghimj(index,1249);
        a = - W_132/ Ghimj(index,1262);
        W_132 = -a;
        W_133 = W_133+ a *Ghimj(index,1263);
        W_134 = W_134+ a *Ghimj(index,1264);
        W_135 = W_135+ a *Ghimj(index,1265);
        W_136 = W_136+ a *Ghimj(index,1266);
        W_137 = W_137+ a *Ghimj(index,1267);
        W_138 = W_138+ a *Ghimj(index,1268);
        Ghimj(index,1269) = W_59;
        Ghimj(index,1270) = W_60;
        Ghimj(index,1271) = W_70;
        Ghimj(index,1272) = W_76;
        Ghimj(index,1273) = W_84;
        Ghimj(index,1274) = W_87;
        Ghimj(index,1275) = W_92;
        Ghimj(index,1276) = W_93;
        Ghimj(index,1277) = W_94;
        Ghimj(index,1278) = W_99;
        Ghimj(index,1279) = W_102;
        Ghimj(index,1280) = W_109;
        Ghimj(index,1281) = W_111;
        Ghimj(index,1282) = W_113;
        Ghimj(index,1283) = W_115;
        Ghimj(index,1284) = W_117;
        Ghimj(index,1285) = W_120;
        Ghimj(index,1286) = W_121;
        Ghimj(index,1287) = W_122;
        Ghimj(index,1288) = W_124;
        Ghimj(index,1289) = W_125;
        Ghimj(index,1290) = W_126;
        Ghimj(index,1291) = W_127;
        Ghimj(index,1292) = W_128;
        Ghimj(index,1293) = W_129;
        Ghimj(index,1294) = W_130;
        Ghimj(index,1295) = W_131;
        Ghimj(index,1296) = W_132;
        Ghimj(index,1297) = W_133;
        Ghimj(index,1298) = W_134;
        Ghimj(index,1299) = W_135;
        Ghimj(index,1300) = W_136;
        Ghimj(index,1301) = W_137;
        Ghimj(index,1302) = W_138;
        W_39 = Ghimj(index,1303);
        W_41 = Ghimj(index,1304);
        W_42 = Ghimj(index,1305);
        W_43 = Ghimj(index,1306);
        W_51 = Ghimj(index,1307);
        W_75 = Ghimj(index,1308);
        W_112 = Ghimj(index,1309);
        W_116 = Ghimj(index,1310);
        W_120 = Ghimj(index,1311);
        W_122 = Ghimj(index,1312);
        W_123 = Ghimj(index,1313);
        W_124 = Ghimj(index,1314);
        W_125 = Ghimj(index,1315);
        W_126 = Ghimj(index,1316);
        W_127 = Ghimj(index,1317);
        W_128 = Ghimj(index,1318);
        W_129 = Ghimj(index,1319);
        W_130 = Ghimj(index,1320);
        W_131 = Ghimj(index,1321);
        W_132 = Ghimj(index,1322);
        W_133 = Ghimj(index,1323);
        W_134 = Ghimj(index,1324);
        W_135 = Ghimj(index,1325);
        W_136 = Ghimj(index,1326);
        W_137 = Ghimj(index,1327);
        W_138 = Ghimj(index,1328);
        a = - W_39/ Ghimj(index,258);
        W_39 = -a;
        W_134 = W_134+ a *Ghimj(index,259);
        a = - W_41/ Ghimj(index,262);
        W_41 = -a;
        W_120 = W_120+ a *Ghimj(index,263);
        a = - W_42/ Ghimj(index,264);
        W_42 = -a;
        W_120 = W_120+ a *Ghimj(index,265);
        a = - W_43/ Ghimj(index,266);
        W_43 = -a;
        W_120 = W_120+ a *Ghimj(index,267);
        a = - W_51/ Ghimj(index,285);
        W_51 = -a;
        W_132 = W_132+ a *Ghimj(index,286);
        W_134 = W_134+ a *Ghimj(index,287);
        a = - W_75/ Ghimj(index,374);
        W_75 = -a;
        W_120 = W_120+ a *Ghimj(index,375);
        W_126 = W_126+ a *Ghimj(index,376);
        a = - W_112/ Ghimj(index,677);
        W_112 = -a;
        W_116 = W_116+ a *Ghimj(index,678);
        W_123 = W_123+ a *Ghimj(index,679);
        W_126 = W_126+ a *Ghimj(index,680);
        W_128 = W_128+ a *Ghimj(index,681);
        W_134 = W_134+ a *Ghimj(index,682);
        W_137 = W_137+ a *Ghimj(index,683);
        W_138 = W_138+ a *Ghimj(index,684);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        a = - W_130/ Ghimj(index,1218);
        W_130 = -a;
        W_131 = W_131+ a *Ghimj(index,1219);
        W_132 = W_132+ a *Ghimj(index,1220);
        W_133 = W_133+ a *Ghimj(index,1221);
        W_134 = W_134+ a *Ghimj(index,1222);
        W_135 = W_135+ a *Ghimj(index,1223);
        W_136 = W_136+ a *Ghimj(index,1224);
        W_137 = W_137+ a *Ghimj(index,1225);
        W_138 = W_138+ a *Ghimj(index,1226);
        a = - W_131/ Ghimj(index,1242);
        W_131 = -a;
        W_132 = W_132+ a *Ghimj(index,1243);
        W_133 = W_133+ a *Ghimj(index,1244);
        W_134 = W_134+ a *Ghimj(index,1245);
        W_135 = W_135+ a *Ghimj(index,1246);
        W_136 = W_136+ a *Ghimj(index,1247);
        W_137 = W_137+ a *Ghimj(index,1248);
        W_138 = W_138+ a *Ghimj(index,1249);
        a = - W_132/ Ghimj(index,1262);
        W_132 = -a;
        W_133 = W_133+ a *Ghimj(index,1263);
        W_134 = W_134+ a *Ghimj(index,1264);
        W_135 = W_135+ a *Ghimj(index,1265);
        W_136 = W_136+ a *Ghimj(index,1266);
        W_137 = W_137+ a *Ghimj(index,1267);
        W_138 = W_138+ a *Ghimj(index,1268);
        a = - W_133/ Ghimj(index,1297);
        W_133 = -a;
        W_134 = W_134+ a *Ghimj(index,1298);
        W_135 = W_135+ a *Ghimj(index,1299);
        W_136 = W_136+ a *Ghimj(index,1300);
        W_137 = W_137+ a *Ghimj(index,1301);
        W_138 = W_138+ a *Ghimj(index,1302);
        Ghimj(index,1303) = W_39;
        Ghimj(index,1304) = W_41;
        Ghimj(index,1305) = W_42;
        Ghimj(index,1306) = W_43;
        Ghimj(index,1307) = W_51;
        Ghimj(index,1308) = W_75;
        Ghimj(index,1309) = W_112;
        Ghimj(index,1310) = W_116;
        Ghimj(index,1311) = W_120;
        Ghimj(index,1312) = W_122;
        Ghimj(index,1313) = W_123;
        Ghimj(index,1314) = W_124;
        Ghimj(index,1315) = W_125;
        Ghimj(index,1316) = W_126;
        Ghimj(index,1317) = W_127;
        Ghimj(index,1318) = W_128;
        Ghimj(index,1319) = W_129;
        Ghimj(index,1320) = W_130;
        Ghimj(index,1321) = W_131;
        Ghimj(index,1322) = W_132;
        Ghimj(index,1323) = W_133;
        Ghimj(index,1324) = W_134;
        Ghimj(index,1325) = W_135;
        Ghimj(index,1326) = W_136;
        Ghimj(index,1327) = W_137;
        Ghimj(index,1328) = W_138;
        W_0 = Ghimj(index,1329);
        W_50 = Ghimj(index,1330);
        W_58 = Ghimj(index,1331);
        W_59 = Ghimj(index,1332);
        W_62 = Ghimj(index,1333);
        W_64 = Ghimj(index,1334);
        W_73 = Ghimj(index,1335);
        W_76 = Ghimj(index,1336);
        W_77 = Ghimj(index,1337);
        W_83 = Ghimj(index,1338);
        W_87 = Ghimj(index,1339);
        W_91 = Ghimj(index,1340);
        W_92 = Ghimj(index,1341);
        W_93 = Ghimj(index,1342);
        W_94 = Ghimj(index,1343);
        W_99 = Ghimj(index,1344);
        W_101 = Ghimj(index,1345);
        W_102 = Ghimj(index,1346);
        W_105 = Ghimj(index,1347);
        W_106 = Ghimj(index,1348);
        W_109 = Ghimj(index,1349);
        W_111 = Ghimj(index,1350);
        W_113 = Ghimj(index,1351);
        W_114 = Ghimj(index,1352);
        W_115 = Ghimj(index,1353);
        W_116 = Ghimj(index,1354);
        W_117 = Ghimj(index,1355);
        W_119 = Ghimj(index,1356);
        W_121 = Ghimj(index,1357);
        W_123 = Ghimj(index,1358);
        W_124 = Ghimj(index,1359);
        W_125 = Ghimj(index,1360);
        W_126 = Ghimj(index,1361);
        W_127 = Ghimj(index,1362);
        W_128 = Ghimj(index,1363);
        W_129 = Ghimj(index,1364);
        W_130 = Ghimj(index,1365);
        W_131 = Ghimj(index,1366);
        W_132 = Ghimj(index,1367);
        W_133 = Ghimj(index,1368);
        W_134 = Ghimj(index,1369);
        W_135 = Ghimj(index,1370);
        W_136 = Ghimj(index,1371);
        W_137 = Ghimj(index,1372);
        W_138 = Ghimj(index,1373);
        a = - W_0/ Ghimj(index,0);
        W_0 = -a;
        a = - W_50/ Ghimj(index,282);
        W_50 = -a;
        W_83 = W_83+ a *Ghimj(index,283);
        W_138 = W_138+ a *Ghimj(index,284);
        a = - W_58/ Ghimj(index,303);
        W_58 = -a;
        W_91 = W_91+ a *Ghimj(index,304);
        W_126 = W_126+ a *Ghimj(index,305);
        a = - W_59/ Ghimj(index,306);
        W_59 = -a;
        W_133 = W_133+ a *Ghimj(index,307);
        W_135 = W_135+ a *Ghimj(index,308);
        a = - W_62/ Ghimj(index,319);
        W_62 = -a;
        W_93 = W_93+ a *Ghimj(index,320);
        W_126 = W_126+ a *Ghimj(index,321);
        W_133 = W_133+ a *Ghimj(index,322);
        a = - W_64/ Ghimj(index,327);
        W_64 = -a;
        W_113 = W_113+ a *Ghimj(index,328);
        W_126 = W_126+ a *Ghimj(index,329);
        W_135 = W_135+ a *Ghimj(index,330);
        a = - W_73/ Ghimj(index,364);
        W_73 = -a;
        W_126 = W_126+ a *Ghimj(index,365);
        W_135 = W_135+ a *Ghimj(index,366);
        W_137 = W_137+ a *Ghimj(index,367);
        a = - W_76/ Ghimj(index,377);
        W_76 = -a;
        W_87 = W_87+ a *Ghimj(index,378);
        W_126 = W_126+ a *Ghimj(index,379);
        W_133 = W_133+ a *Ghimj(index,380);
        W_135 = W_135+ a *Ghimj(index,381);
        a = - W_77/ Ghimj(index,382);
        W_77 = -a;
        W_121 = W_121+ a *Ghimj(index,383);
        W_126 = W_126+ a *Ghimj(index,384);
        W_135 = W_135+ a *Ghimj(index,385);
        a = - W_83/ Ghimj(index,416);
        W_83 = -a;
        W_128 = W_128+ a *Ghimj(index,417);
        W_135 = W_135+ a *Ghimj(index,418);
        W_136 = W_136+ a *Ghimj(index,419);
        W_138 = W_138+ a *Ghimj(index,420);
        a = - W_87/ Ghimj(index,444);
        W_87 = -a;
        W_92 = W_92+ a *Ghimj(index,445);
        W_124 = W_124+ a *Ghimj(index,446);
        W_126 = W_126+ a *Ghimj(index,447);
        W_135 = W_135+ a *Ghimj(index,448);
        W_137 = W_137+ a *Ghimj(index,449);
        a = - W_91/ Ghimj(index,481);
        W_91 = -a;
        W_106 = W_106+ a *Ghimj(index,482);
        W_109 = W_109+ a *Ghimj(index,483);
        W_126 = W_126+ a *Ghimj(index,484);
        W_133 = W_133+ a *Ghimj(index,485);
        W_136 = W_136+ a *Ghimj(index,486);
        a = - W_92/ Ghimj(index,489);
        W_92 = -a;
        W_124 = W_124+ a *Ghimj(index,490);
        W_126 = W_126+ a *Ghimj(index,491);
        W_133 = W_133+ a *Ghimj(index,492);
        W_135 = W_135+ a *Ghimj(index,493);
        W_137 = W_137+ a *Ghimj(index,494);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        a = - W_99/ Ghimj(index,565);
        W_99 = -a;
        W_102 = W_102+ a *Ghimj(index,566);
        W_111 = W_111+ a *Ghimj(index,567);
        W_125 = W_125+ a *Ghimj(index,568);
        W_126 = W_126+ a *Ghimj(index,569);
        W_133 = W_133+ a *Ghimj(index,570);
        W_137 = W_137+ a *Ghimj(index,571);
        a = - W_101/ Ghimj(index,586);
        W_101 = -a;
        W_105 = W_105+ a *Ghimj(index,587);
        W_114 = W_114+ a *Ghimj(index,588);
        W_116 = W_116+ a *Ghimj(index,589);
        W_119 = W_119+ a *Ghimj(index,590);
        W_123 = W_123+ a *Ghimj(index,591);
        W_126 = W_126+ a *Ghimj(index,592);
        W_128 = W_128+ a *Ghimj(index,593);
        W_130 = W_130+ a *Ghimj(index,594);
        W_135 = W_135+ a *Ghimj(index,595);
        W_136 = W_136+ a *Ghimj(index,596);
        W_138 = W_138+ a *Ghimj(index,597);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        a = - W_111/ Ghimj(index,669);
        W_111 = -a;
        W_115 = W_115+ a *Ghimj(index,670);
        W_124 = W_124+ a *Ghimj(index,671);
        W_125 = W_125+ a *Ghimj(index,672);
        W_126 = W_126+ a *Ghimj(index,673);
        W_133 = W_133+ a *Ghimj(index,674);
        W_136 = W_136+ a *Ghimj(index,675);
        W_137 = W_137+ a *Ghimj(index,676);
        a = - W_113/ Ghimj(index,689);
        W_113 = -a;
        W_124 = W_124+ a *Ghimj(index,690);
        W_125 = W_125+ a *Ghimj(index,691);
        W_126 = W_126+ a *Ghimj(index,692);
        W_133 = W_133+ a *Ghimj(index,693);
        W_135 = W_135+ a *Ghimj(index,694);
        W_136 = W_136+ a *Ghimj(index,695);
        W_137 = W_137+ a *Ghimj(index,696);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        a = - W_130/ Ghimj(index,1218);
        W_130 = -a;
        W_131 = W_131+ a *Ghimj(index,1219);
        W_132 = W_132+ a *Ghimj(index,1220);
        W_133 = W_133+ a *Ghimj(index,1221);
        W_134 = W_134+ a *Ghimj(index,1222);
        W_135 = W_135+ a *Ghimj(index,1223);
        W_136 = W_136+ a *Ghimj(index,1224);
        W_137 = W_137+ a *Ghimj(index,1225);
        W_138 = W_138+ a *Ghimj(index,1226);
        a = - W_131/ Ghimj(index,1242);
        W_131 = -a;
        W_132 = W_132+ a *Ghimj(index,1243);
        W_133 = W_133+ a *Ghimj(index,1244);
        W_134 = W_134+ a *Ghimj(index,1245);
        W_135 = W_135+ a *Ghimj(index,1246);
        W_136 = W_136+ a *Ghimj(index,1247);
        W_137 = W_137+ a *Ghimj(index,1248);
        W_138 = W_138+ a *Ghimj(index,1249);
        a = - W_132/ Ghimj(index,1262);
        W_132 = -a;
        W_133 = W_133+ a *Ghimj(index,1263);
        W_134 = W_134+ a *Ghimj(index,1264);
        W_135 = W_135+ a *Ghimj(index,1265);
        W_136 = W_136+ a *Ghimj(index,1266);
        W_137 = W_137+ a *Ghimj(index,1267);
        W_138 = W_138+ a *Ghimj(index,1268);
        a = - W_133/ Ghimj(index,1297);
        W_133 = -a;
        W_134 = W_134+ a *Ghimj(index,1298);
        W_135 = W_135+ a *Ghimj(index,1299);
        W_136 = W_136+ a *Ghimj(index,1300);
        W_137 = W_137+ a *Ghimj(index,1301);
        W_138 = W_138+ a *Ghimj(index,1302);
        a = - W_134/ Ghimj(index,1324);
        W_134 = -a;
        W_135 = W_135+ a *Ghimj(index,1325);
        W_136 = W_136+ a *Ghimj(index,1326);
        W_137 = W_137+ a *Ghimj(index,1327);
        W_138 = W_138+ a *Ghimj(index,1328);
        Ghimj(index,1329) = W_0;
        Ghimj(index,1330) = W_50;
        Ghimj(index,1331) = W_58;
        Ghimj(index,1332) = W_59;
        Ghimj(index,1333) = W_62;
        Ghimj(index,1334) = W_64;
        Ghimj(index,1335) = W_73;
        Ghimj(index,1336) = W_76;
        Ghimj(index,1337) = W_77;
        Ghimj(index,1338) = W_83;
        Ghimj(index,1339) = W_87;
        Ghimj(index,1340) = W_91;
        Ghimj(index,1341) = W_92;
        Ghimj(index,1342) = W_93;
        Ghimj(index,1343) = W_94;
        Ghimj(index,1344) = W_99;
        Ghimj(index,1345) = W_101;
        Ghimj(index,1346) = W_102;
        Ghimj(index,1347) = W_105;
        Ghimj(index,1348) = W_106;
        Ghimj(index,1349) = W_109;
        Ghimj(index,1350) = W_111;
        Ghimj(index,1351) = W_113;
        Ghimj(index,1352) = W_114;
        Ghimj(index,1353) = W_115;
        Ghimj(index,1354) = W_116;
        Ghimj(index,1355) = W_117;
        Ghimj(index,1356) = W_119;
        Ghimj(index,1357) = W_121;
        Ghimj(index,1358) = W_123;
        Ghimj(index,1359) = W_124;
        Ghimj(index,1360) = W_125;
        Ghimj(index,1361) = W_126;
        Ghimj(index,1362) = W_127;
        Ghimj(index,1363) = W_128;
        Ghimj(index,1364) = W_129;
        Ghimj(index,1365) = W_130;
        Ghimj(index,1366) = W_131;
        Ghimj(index,1367) = W_132;
        Ghimj(index,1368) = W_133;
        Ghimj(index,1369) = W_134;
        Ghimj(index,1370) = W_135;
        Ghimj(index,1371) = W_136;
        Ghimj(index,1372) = W_137;
        Ghimj(index,1373) = W_138;
        W_73 = Ghimj(index,1374);
        W_83 = Ghimj(index,1375);
        W_101 = Ghimj(index,1376);
        W_105 = Ghimj(index,1377);
        W_106 = Ghimj(index,1378);
        W_107 = Ghimj(index,1379);
        W_114 = Ghimj(index,1380);
        W_116 = Ghimj(index,1381);
        W_117 = Ghimj(index,1382);
        W_119 = Ghimj(index,1383);
        W_121 = Ghimj(index,1384);
        W_123 = Ghimj(index,1385);
        W_124 = Ghimj(index,1386);
        W_125 = Ghimj(index,1387);
        W_126 = Ghimj(index,1388);
        W_127 = Ghimj(index,1389);
        W_128 = Ghimj(index,1390);
        W_129 = Ghimj(index,1391);
        W_130 = Ghimj(index,1392);
        W_131 = Ghimj(index,1393);
        W_132 = Ghimj(index,1394);
        W_133 = Ghimj(index,1395);
        W_134 = Ghimj(index,1396);
        W_135 = Ghimj(index,1397);
        W_136 = Ghimj(index,1398);
        W_137 = Ghimj(index,1399);
        W_138 = Ghimj(index,1400);
        a = - W_73/ Ghimj(index,364);
        W_73 = -a;
        W_126 = W_126+ a *Ghimj(index,365);
        W_135 = W_135+ a *Ghimj(index,366);
        W_137 = W_137+ a *Ghimj(index,367);
        a = - W_83/ Ghimj(index,416);
        W_83 = -a;
        W_128 = W_128+ a *Ghimj(index,417);
        W_135 = W_135+ a *Ghimj(index,418);
        W_136 = W_136+ a *Ghimj(index,419);
        W_138 = W_138+ a *Ghimj(index,420);
        a = - W_101/ Ghimj(index,586);
        W_101 = -a;
        W_105 = W_105+ a *Ghimj(index,587);
        W_114 = W_114+ a *Ghimj(index,588);
        W_116 = W_116+ a *Ghimj(index,589);
        W_119 = W_119+ a *Ghimj(index,590);
        W_123 = W_123+ a *Ghimj(index,591);
        W_126 = W_126+ a *Ghimj(index,592);
        W_128 = W_128+ a *Ghimj(index,593);
        W_130 = W_130+ a *Ghimj(index,594);
        W_135 = W_135+ a *Ghimj(index,595);
        W_136 = W_136+ a *Ghimj(index,596);
        W_138 = W_138+ a *Ghimj(index,597);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        a = - W_130/ Ghimj(index,1218);
        W_130 = -a;
        W_131 = W_131+ a *Ghimj(index,1219);
        W_132 = W_132+ a *Ghimj(index,1220);
        W_133 = W_133+ a *Ghimj(index,1221);
        W_134 = W_134+ a *Ghimj(index,1222);
        W_135 = W_135+ a *Ghimj(index,1223);
        W_136 = W_136+ a *Ghimj(index,1224);
        W_137 = W_137+ a *Ghimj(index,1225);
        W_138 = W_138+ a *Ghimj(index,1226);
        a = - W_131/ Ghimj(index,1242);
        W_131 = -a;
        W_132 = W_132+ a *Ghimj(index,1243);
        W_133 = W_133+ a *Ghimj(index,1244);
        W_134 = W_134+ a *Ghimj(index,1245);
        W_135 = W_135+ a *Ghimj(index,1246);
        W_136 = W_136+ a *Ghimj(index,1247);
        W_137 = W_137+ a *Ghimj(index,1248);
        W_138 = W_138+ a *Ghimj(index,1249);
        a = - W_132/ Ghimj(index,1262);
        W_132 = -a;
        W_133 = W_133+ a *Ghimj(index,1263);
        W_134 = W_134+ a *Ghimj(index,1264);
        W_135 = W_135+ a *Ghimj(index,1265);
        W_136 = W_136+ a *Ghimj(index,1266);
        W_137 = W_137+ a *Ghimj(index,1267);
        W_138 = W_138+ a *Ghimj(index,1268);
        a = - W_133/ Ghimj(index,1297);
        W_133 = -a;
        W_134 = W_134+ a *Ghimj(index,1298);
        W_135 = W_135+ a *Ghimj(index,1299);
        W_136 = W_136+ a *Ghimj(index,1300);
        W_137 = W_137+ a *Ghimj(index,1301);
        W_138 = W_138+ a *Ghimj(index,1302);
        a = - W_134/ Ghimj(index,1324);
        W_134 = -a;
        W_135 = W_135+ a *Ghimj(index,1325);
        W_136 = W_136+ a *Ghimj(index,1326);
        W_137 = W_137+ a *Ghimj(index,1327);
        W_138 = W_138+ a *Ghimj(index,1328);
        a = - W_135/ Ghimj(index,1370);
        W_135 = -a;
        W_136 = W_136+ a *Ghimj(index,1371);
        W_137 = W_137+ a *Ghimj(index,1372);
        W_138 = W_138+ a *Ghimj(index,1373);
        Ghimj(index,1374) = W_73;
        Ghimj(index,1375) = W_83;
        Ghimj(index,1376) = W_101;
        Ghimj(index,1377) = W_105;
        Ghimj(index,1378) = W_106;
        Ghimj(index,1379) = W_107;
        Ghimj(index,1380) = W_114;
        Ghimj(index,1381) = W_116;
        Ghimj(index,1382) = W_117;
        Ghimj(index,1383) = W_119;
        Ghimj(index,1384) = W_121;
        Ghimj(index,1385) = W_123;
        Ghimj(index,1386) = W_124;
        Ghimj(index,1387) = W_125;
        Ghimj(index,1388) = W_126;
        Ghimj(index,1389) = W_127;
        Ghimj(index,1390) = W_128;
        Ghimj(index,1391) = W_129;
        Ghimj(index,1392) = W_130;
        Ghimj(index,1393) = W_131;
        Ghimj(index,1394) = W_132;
        Ghimj(index,1395) = W_133;
        Ghimj(index,1396) = W_134;
        Ghimj(index,1397) = W_135;
        Ghimj(index,1398) = W_136;
        Ghimj(index,1399) = W_137;
        Ghimj(index,1400) = W_138;
        W_46 = Ghimj(index,1401);
        W_56 = Ghimj(index,1402);
        W_62 = Ghimj(index,1403);
        W_65 = Ghimj(index,1404);
        W_66 = Ghimj(index,1405);
        W_69 = Ghimj(index,1406);
        W_71 = Ghimj(index,1407);
        W_73 = Ghimj(index,1408);
        W_78 = Ghimj(index,1409);
        W_79 = Ghimj(index,1410);
        W_81 = Ghimj(index,1411);
        W_82 = Ghimj(index,1412);
        W_87 = Ghimj(index,1413);
        W_88 = Ghimj(index,1414);
        W_89 = Ghimj(index,1415);
        W_91 = Ghimj(index,1416);
        W_92 = Ghimj(index,1417);
        W_93 = Ghimj(index,1418);
        W_94 = Ghimj(index,1419);
        W_96 = Ghimj(index,1420);
        W_99 = Ghimj(index,1421);
        W_102 = Ghimj(index,1422);
        W_103 = Ghimj(index,1423);
        W_104 = Ghimj(index,1424);
        W_106 = Ghimj(index,1425);
        W_107 = Ghimj(index,1426);
        W_108 = Ghimj(index,1427);
        W_109 = Ghimj(index,1428);
        W_110 = Ghimj(index,1429);
        W_111 = Ghimj(index,1430);
        W_113 = Ghimj(index,1431);
        W_114 = Ghimj(index,1432);
        W_115 = Ghimj(index,1433);
        W_117 = Ghimj(index,1434);
        W_119 = Ghimj(index,1435);
        W_121 = Ghimj(index,1436);
        W_122 = Ghimj(index,1437);
        W_124 = Ghimj(index,1438);
        W_125 = Ghimj(index,1439);
        W_126 = Ghimj(index,1440);
        W_127 = Ghimj(index,1441);
        W_128 = Ghimj(index,1442);
        W_129 = Ghimj(index,1443);
        W_130 = Ghimj(index,1444);
        W_131 = Ghimj(index,1445);
        W_132 = Ghimj(index,1446);
        W_133 = Ghimj(index,1447);
        W_134 = Ghimj(index,1448);
        W_135 = Ghimj(index,1449);
        W_136 = Ghimj(index,1450);
        W_137 = Ghimj(index,1451);
        W_138 = Ghimj(index,1452);
        a = - W_46/ Ghimj(index,272);
        W_46 = -a;
        W_81 = W_81+ a *Ghimj(index,273);
        W_124 = W_124+ a *Ghimj(index,274);
        W_137 = W_137+ a *Ghimj(index,275);
        a = - W_56/ Ghimj(index,296);
        W_56 = -a;
        W_65 = W_65+ a *Ghimj(index,297);
        W_81 = W_81+ a *Ghimj(index,298);
        W_126 = W_126+ a *Ghimj(index,299);
        a = - W_62/ Ghimj(index,319);
        W_62 = -a;
        W_93 = W_93+ a *Ghimj(index,320);
        W_126 = W_126+ a *Ghimj(index,321);
        W_133 = W_133+ a *Ghimj(index,322);
        a = - W_65/ Ghimj(index,331);
        W_65 = -a;
        W_114 = W_114+ a *Ghimj(index,332);
        W_126 = W_126+ a *Ghimj(index,333);
        W_132 = W_132+ a *Ghimj(index,334);
        a = - W_66/ Ghimj(index,335);
        W_66 = -a;
        W_109 = W_109+ a *Ghimj(index,336);
        W_126 = W_126+ a *Ghimj(index,337);
        W_137 = W_137+ a *Ghimj(index,338);
        a = - W_69/ Ghimj(index,347);
        W_69 = -a;
        W_93 = W_93+ a *Ghimj(index,348);
        W_126 = W_126+ a *Ghimj(index,349);
        W_137 = W_137+ a *Ghimj(index,350);
        a = - W_71/ Ghimj(index,356);
        W_71 = -a;
        W_117 = W_117+ a *Ghimj(index,357);
        W_126 = W_126+ a *Ghimj(index,358);
        W_137 = W_137+ a *Ghimj(index,359);
        a = - W_73/ Ghimj(index,364);
        W_73 = -a;
        W_126 = W_126+ a *Ghimj(index,365);
        W_135 = W_135+ a *Ghimj(index,366);
        W_137 = W_137+ a *Ghimj(index,367);
        a = - W_78/ Ghimj(index,386);
        W_78 = -a;
        W_103 = W_103+ a *Ghimj(index,387);
        W_106 = W_106+ a *Ghimj(index,388);
        W_107 = W_107+ a *Ghimj(index,389);
        W_110 = W_110+ a *Ghimj(index,390);
        W_124 = W_124+ a *Ghimj(index,391);
        W_126 = W_126+ a *Ghimj(index,392);
        a = - W_79/ Ghimj(index,393);
        W_79 = -a;
        W_102 = W_102+ a *Ghimj(index,394);
        W_126 = W_126+ a *Ghimj(index,395);
        W_137 = W_137+ a *Ghimj(index,396);
        a = - W_81/ Ghimj(index,405);
        W_81 = -a;
        W_114 = W_114+ a *Ghimj(index,406);
        W_124 = W_124+ a *Ghimj(index,407);
        W_126 = W_126+ a *Ghimj(index,408);
        W_127 = W_127+ a *Ghimj(index,409);
        W_129 = W_129+ a *Ghimj(index,410);
        W_136 = W_136+ a *Ghimj(index,411);
        a = - W_82/ Ghimj(index,412);
        W_82 = -a;
        W_113 = W_113+ a *Ghimj(index,413);
        W_126 = W_126+ a *Ghimj(index,414);
        W_137 = W_137+ a *Ghimj(index,415);
        a = - W_87/ Ghimj(index,444);
        W_87 = -a;
        W_92 = W_92+ a *Ghimj(index,445);
        W_124 = W_124+ a *Ghimj(index,446);
        W_126 = W_126+ a *Ghimj(index,447);
        W_135 = W_135+ a *Ghimj(index,448);
        W_137 = W_137+ a *Ghimj(index,449);
        a = - W_88/ Ghimj(index,450);
        W_88 = -a;
        W_103 = W_103+ a *Ghimj(index,451);
        W_106 = W_106+ a *Ghimj(index,452);
        W_124 = W_124+ a *Ghimj(index,453);
        W_126 = W_126+ a *Ghimj(index,454);
        W_127 = W_127+ a *Ghimj(index,455);
        W_137 = W_137+ a *Ghimj(index,456);
        a = - W_89/ Ghimj(index,457);
        W_89 = -a;
        W_93 = W_93+ a *Ghimj(index,458);
        W_94 = W_94+ a *Ghimj(index,459);
        W_102 = W_102+ a *Ghimj(index,460);
        W_107 = W_107+ a *Ghimj(index,461);
        W_109 = W_109+ a *Ghimj(index,462);
        W_113 = W_113+ a *Ghimj(index,463);
        W_117 = W_117+ a *Ghimj(index,464);
        W_124 = W_124+ a *Ghimj(index,465);
        W_125 = W_125+ a *Ghimj(index,466);
        W_126 = W_126+ a *Ghimj(index,467);
        a = - W_91/ Ghimj(index,481);
        W_91 = -a;
        W_106 = W_106+ a *Ghimj(index,482);
        W_109 = W_109+ a *Ghimj(index,483);
        W_126 = W_126+ a *Ghimj(index,484);
        W_133 = W_133+ a *Ghimj(index,485);
        W_136 = W_136+ a *Ghimj(index,486);
        a = - W_92/ Ghimj(index,489);
        W_92 = -a;
        W_124 = W_124+ a *Ghimj(index,490);
        W_126 = W_126+ a *Ghimj(index,491);
        W_133 = W_133+ a *Ghimj(index,492);
        W_135 = W_135+ a *Ghimj(index,493);
        W_137 = W_137+ a *Ghimj(index,494);
        a = - W_93/ Ghimj(index,497);
        W_93 = -a;
        W_125 = W_125+ a *Ghimj(index,498);
        W_126 = W_126+ a *Ghimj(index,499);
        W_133 = W_133+ a *Ghimj(index,500);
        W_137 = W_137+ a *Ghimj(index,501);
        a = - W_94/ Ghimj(index,505);
        W_94 = -a;
        W_125 = W_125+ a *Ghimj(index,506);
        W_126 = W_126+ a *Ghimj(index,507);
        W_133 = W_133+ a *Ghimj(index,508);
        W_137 = W_137+ a *Ghimj(index,509);
        a = - W_96/ Ghimj(index,538);
        W_96 = -a;
        W_107 = W_107+ a *Ghimj(index,539);
        W_108 = W_108+ a *Ghimj(index,540);
        W_109 = W_109+ a *Ghimj(index,541);
        W_110 = W_110+ a *Ghimj(index,542);
        W_113 = W_113+ a *Ghimj(index,543);
        W_124 = W_124+ a *Ghimj(index,544);
        W_125 = W_125+ a *Ghimj(index,545);
        W_126 = W_126+ a *Ghimj(index,546);
        W_133 = W_133+ a *Ghimj(index,547);
        W_137 = W_137+ a *Ghimj(index,548);
        a = - W_99/ Ghimj(index,565);
        W_99 = -a;
        W_102 = W_102+ a *Ghimj(index,566);
        W_111 = W_111+ a *Ghimj(index,567);
        W_125 = W_125+ a *Ghimj(index,568);
        W_126 = W_126+ a *Ghimj(index,569);
        W_133 = W_133+ a *Ghimj(index,570);
        W_137 = W_137+ a *Ghimj(index,571);
        a = - W_102/ Ghimj(index,600);
        W_102 = -a;
        W_125 = W_125+ a *Ghimj(index,601);
        W_126 = W_126+ a *Ghimj(index,602);
        W_133 = W_133+ a *Ghimj(index,603);
        W_137 = W_137+ a *Ghimj(index,604);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_108/ Ghimj(index,636);
        W_108 = -a;
        W_109 = W_109+ a *Ghimj(index,637);
        W_113 = W_113+ a *Ghimj(index,638);
        W_115 = W_115+ a *Ghimj(index,639);
        W_124 = W_124+ a *Ghimj(index,640);
        W_125 = W_125+ a *Ghimj(index,641);
        W_126 = W_126+ a *Ghimj(index,642);
        W_133 = W_133+ a *Ghimj(index,643);
        W_135 = W_135+ a *Ghimj(index,644);
        W_136 = W_136+ a *Ghimj(index,645);
        W_137 = W_137+ a *Ghimj(index,646);
        a = - W_109/ Ghimj(index,648);
        W_109 = -a;
        W_124 = W_124+ a *Ghimj(index,649);
        W_125 = W_125+ a *Ghimj(index,650);
        W_126 = W_126+ a *Ghimj(index,651);
        W_133 = W_133+ a *Ghimj(index,652);
        W_136 = W_136+ a *Ghimj(index,653);
        W_137 = W_137+ a *Ghimj(index,654);
        a = - W_110/ Ghimj(index,659);
        W_110 = -a;
        W_124 = W_124+ a *Ghimj(index,660);
        W_125 = W_125+ a *Ghimj(index,661);
        W_126 = W_126+ a *Ghimj(index,662);
        W_133 = W_133+ a *Ghimj(index,663);
        W_136 = W_136+ a *Ghimj(index,664);
        W_137 = W_137+ a *Ghimj(index,665);
        a = - W_111/ Ghimj(index,669);
        W_111 = -a;
        W_115 = W_115+ a *Ghimj(index,670);
        W_124 = W_124+ a *Ghimj(index,671);
        W_125 = W_125+ a *Ghimj(index,672);
        W_126 = W_126+ a *Ghimj(index,673);
        W_133 = W_133+ a *Ghimj(index,674);
        W_136 = W_136+ a *Ghimj(index,675);
        W_137 = W_137+ a *Ghimj(index,676);
        a = - W_113/ Ghimj(index,689);
        W_113 = -a;
        W_124 = W_124+ a *Ghimj(index,690);
        W_125 = W_125+ a *Ghimj(index,691);
        W_126 = W_126+ a *Ghimj(index,692);
        W_133 = W_133+ a *Ghimj(index,693);
        W_135 = W_135+ a *Ghimj(index,694);
        W_136 = W_136+ a *Ghimj(index,695);
        W_137 = W_137+ a *Ghimj(index,696);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_115/ Ghimj(index,706);
        W_115 = -a;
        W_124 = W_124+ a *Ghimj(index,707);
        W_126 = W_126+ a *Ghimj(index,708);
        W_127 = W_127+ a *Ghimj(index,709);
        W_129 = W_129+ a *Ghimj(index,710);
        W_133 = W_133+ a *Ghimj(index,711);
        W_136 = W_136+ a *Ghimj(index,712);
        W_137 = W_137+ a *Ghimj(index,713);
        a = - W_117/ Ghimj(index,731);
        W_117 = -a;
        W_121 = W_121+ a *Ghimj(index,732);
        W_124 = W_124+ a *Ghimj(index,733);
        W_125 = W_125+ a *Ghimj(index,734);
        W_126 = W_126+ a *Ghimj(index,735);
        W_127 = W_127+ a *Ghimj(index,736);
        W_129 = W_129+ a *Ghimj(index,737);
        W_133 = W_133+ a *Ghimj(index,738);
        W_136 = W_136+ a *Ghimj(index,739);
        W_137 = W_137+ a *Ghimj(index,740);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        a = - W_130/ Ghimj(index,1218);
        W_130 = -a;
        W_131 = W_131+ a *Ghimj(index,1219);
        W_132 = W_132+ a *Ghimj(index,1220);
        W_133 = W_133+ a *Ghimj(index,1221);
        W_134 = W_134+ a *Ghimj(index,1222);
        W_135 = W_135+ a *Ghimj(index,1223);
        W_136 = W_136+ a *Ghimj(index,1224);
        W_137 = W_137+ a *Ghimj(index,1225);
        W_138 = W_138+ a *Ghimj(index,1226);
        a = - W_131/ Ghimj(index,1242);
        W_131 = -a;
        W_132 = W_132+ a *Ghimj(index,1243);
        W_133 = W_133+ a *Ghimj(index,1244);
        W_134 = W_134+ a *Ghimj(index,1245);
        W_135 = W_135+ a *Ghimj(index,1246);
        W_136 = W_136+ a *Ghimj(index,1247);
        W_137 = W_137+ a *Ghimj(index,1248);
        W_138 = W_138+ a *Ghimj(index,1249);
        a = - W_132/ Ghimj(index,1262);
        W_132 = -a;
        W_133 = W_133+ a *Ghimj(index,1263);
        W_134 = W_134+ a *Ghimj(index,1264);
        W_135 = W_135+ a *Ghimj(index,1265);
        W_136 = W_136+ a *Ghimj(index,1266);
        W_137 = W_137+ a *Ghimj(index,1267);
        W_138 = W_138+ a *Ghimj(index,1268);
        a = - W_133/ Ghimj(index,1297);
        W_133 = -a;
        W_134 = W_134+ a *Ghimj(index,1298);
        W_135 = W_135+ a *Ghimj(index,1299);
        W_136 = W_136+ a *Ghimj(index,1300);
        W_137 = W_137+ a *Ghimj(index,1301);
        W_138 = W_138+ a *Ghimj(index,1302);
        a = - W_134/ Ghimj(index,1324);
        W_134 = -a;
        W_135 = W_135+ a *Ghimj(index,1325);
        W_136 = W_136+ a *Ghimj(index,1326);
        W_137 = W_137+ a *Ghimj(index,1327);
        W_138 = W_138+ a *Ghimj(index,1328);
        a = - W_135/ Ghimj(index,1370);
        W_135 = -a;
        W_136 = W_136+ a *Ghimj(index,1371);
        W_137 = W_137+ a *Ghimj(index,1372);
        W_138 = W_138+ a *Ghimj(index,1373);
        a = - W_136/ Ghimj(index,1398);
        W_136 = -a;
        W_137 = W_137+ a *Ghimj(index,1399);
        W_138 = W_138+ a *Ghimj(index,1400);
        Ghimj(index,1401) = W_46;
        Ghimj(index,1402) = W_56;
        Ghimj(index,1403) = W_62;
        Ghimj(index,1404) = W_65;
        Ghimj(index,1405) = W_66;
        Ghimj(index,1406) = W_69;
        Ghimj(index,1407) = W_71;
        Ghimj(index,1408) = W_73;
        Ghimj(index,1409) = W_78;
        Ghimj(index,1410) = W_79;
        Ghimj(index,1411) = W_81;
        Ghimj(index,1412) = W_82;
        Ghimj(index,1413) = W_87;
        Ghimj(index,1414) = W_88;
        Ghimj(index,1415) = W_89;
        Ghimj(index,1416) = W_91;
        Ghimj(index,1417) = W_92;
        Ghimj(index,1418) = W_93;
        Ghimj(index,1419) = W_94;
        Ghimj(index,1420) = W_96;
        Ghimj(index,1421) = W_99;
        Ghimj(index,1422) = W_102;
        Ghimj(index,1423) = W_103;
        Ghimj(index,1424) = W_104;
        Ghimj(index,1425) = W_106;
        Ghimj(index,1426) = W_107;
        Ghimj(index,1427) = W_108;
        Ghimj(index,1428) = W_109;
        Ghimj(index,1429) = W_110;
        Ghimj(index,1430) = W_111;
        Ghimj(index,1431) = W_113;
        Ghimj(index,1432) = W_114;
        Ghimj(index,1433) = W_115;
        Ghimj(index,1434) = W_117;
        Ghimj(index,1435) = W_119;
        Ghimj(index,1436) = W_121;
        Ghimj(index,1437) = W_122;
        Ghimj(index,1438) = W_124;
        Ghimj(index,1439) = W_125;
        Ghimj(index,1440) = W_126;
        Ghimj(index,1441) = W_127;
        Ghimj(index,1442) = W_128;
        Ghimj(index,1443) = W_129;
        Ghimj(index,1444) = W_130;
        Ghimj(index,1445) = W_131;
        Ghimj(index,1446) = W_132;
        Ghimj(index,1447) = W_133;
        Ghimj(index,1448) = W_134;
        Ghimj(index,1449) = W_135;
        Ghimj(index,1450) = W_136;
        Ghimj(index,1451) = W_137;
        Ghimj(index,1452) = W_138;
        W_83 = Ghimj(index,1453);
        W_88 = Ghimj(index,1454);
        W_97 = Ghimj(index,1455);
        W_98 = Ghimj(index,1456);
        W_103 = Ghimj(index,1457);
        W_104 = Ghimj(index,1458);
        W_105 = Ghimj(index,1459);
        W_106 = Ghimj(index,1460);
        W_107 = Ghimj(index,1461);
        W_112 = Ghimj(index,1462);
        W_114 = Ghimj(index,1463);
        W_116 = Ghimj(index,1464);
        W_118 = Ghimj(index,1465);
        W_119 = Ghimj(index,1466);
        W_120 = Ghimj(index,1467);
        W_121 = Ghimj(index,1468);
        W_122 = Ghimj(index,1469);
        W_123 = Ghimj(index,1470);
        W_124 = Ghimj(index,1471);
        W_125 = Ghimj(index,1472);
        W_126 = Ghimj(index,1473);
        W_127 = Ghimj(index,1474);
        W_128 = Ghimj(index,1475);
        W_129 = Ghimj(index,1476);
        W_130 = Ghimj(index,1477);
        W_131 = Ghimj(index,1478);
        W_132 = Ghimj(index,1479);
        W_133 = Ghimj(index,1480);
        W_134 = Ghimj(index,1481);
        W_135 = Ghimj(index,1482);
        W_136 = Ghimj(index,1483);
        W_137 = Ghimj(index,1484);
        W_138 = Ghimj(index,1485);
        a = - W_83/ Ghimj(index,416);
        W_83 = -a;
        W_128 = W_128+ a *Ghimj(index,417);
        W_135 = W_135+ a *Ghimj(index,418);
        W_136 = W_136+ a *Ghimj(index,419);
        W_138 = W_138+ a *Ghimj(index,420);
        a = - W_88/ Ghimj(index,450);
        W_88 = -a;
        W_103 = W_103+ a *Ghimj(index,451);
        W_106 = W_106+ a *Ghimj(index,452);
        W_124 = W_124+ a *Ghimj(index,453);
        W_126 = W_126+ a *Ghimj(index,454);
        W_127 = W_127+ a *Ghimj(index,455);
        W_137 = W_137+ a *Ghimj(index,456);
        a = - W_97/ Ghimj(index,549);
        W_97 = -a;
        W_98 = W_98+ a *Ghimj(index,550);
        W_120 = W_120+ a *Ghimj(index,551);
        W_122 = W_122+ a *Ghimj(index,552);
        W_126 = W_126+ a *Ghimj(index,553);
        W_127 = W_127+ a *Ghimj(index,554);
        W_130 = W_130+ a *Ghimj(index,555);
        W_137 = W_137+ a *Ghimj(index,556);
        a = - W_98/ Ghimj(index,557);
        W_98 = -a;
        W_107 = W_107+ a *Ghimj(index,558);
        W_120 = W_120+ a *Ghimj(index,559);
        W_124 = W_124+ a *Ghimj(index,560);
        W_126 = W_126+ a *Ghimj(index,561);
        W_127 = W_127+ a *Ghimj(index,562);
        a = - W_103/ Ghimj(index,605);
        W_103 = -a;
        W_124 = W_124+ a *Ghimj(index,606);
        W_126 = W_126+ a *Ghimj(index,607);
        W_127 = W_127+ a *Ghimj(index,608);
        W_129 = W_129+ a *Ghimj(index,609);
        a = - W_104/ Ghimj(index,610);
        W_104 = -a;
        W_125 = W_125+ a *Ghimj(index,611);
        W_126 = W_126+ a *Ghimj(index,612);
        W_127 = W_127+ a *Ghimj(index,613);
        W_129 = W_129+ a *Ghimj(index,614);
        W_137 = W_137+ a *Ghimj(index,615);
        a = - W_105/ Ghimj(index,616);
        W_105 = -a;
        W_128 = W_128+ a *Ghimj(index,617);
        W_129 = W_129+ a *Ghimj(index,618);
        W_132 = W_132+ a *Ghimj(index,619);
        W_135 = W_135+ a *Ghimj(index,620);
        W_138 = W_138+ a *Ghimj(index,621);
        a = - W_106/ Ghimj(index,622);
        W_106 = -a;
        W_124 = W_124+ a *Ghimj(index,623);
        W_126 = W_126+ a *Ghimj(index,624);
        W_136 = W_136+ a *Ghimj(index,625);
        a = - W_107/ Ghimj(index,626);
        W_107 = -a;
        W_124 = W_124+ a *Ghimj(index,627);
        W_126 = W_126+ a *Ghimj(index,628);
        W_136 = W_136+ a *Ghimj(index,629);
        a = - W_112/ Ghimj(index,677);
        W_112 = -a;
        W_116 = W_116+ a *Ghimj(index,678);
        W_123 = W_123+ a *Ghimj(index,679);
        W_126 = W_126+ a *Ghimj(index,680);
        W_128 = W_128+ a *Ghimj(index,681);
        W_134 = W_134+ a *Ghimj(index,682);
        W_137 = W_137+ a *Ghimj(index,683);
        W_138 = W_138+ a *Ghimj(index,684);
        a = - W_114/ Ghimj(index,697);
        W_114 = -a;
        W_126 = W_126+ a *Ghimj(index,698);
        W_127 = W_127+ a *Ghimj(index,699);
        W_129 = W_129+ a *Ghimj(index,700);
        W_132 = W_132+ a *Ghimj(index,701);
        W_136 = W_136+ a *Ghimj(index,702);
        a = - W_116/ Ghimj(index,714);
        W_116 = -a;
        W_123 = W_123+ a *Ghimj(index,715);
        W_127 = W_127+ a *Ghimj(index,716);
        W_128 = W_128+ a *Ghimj(index,717);
        W_131 = W_131+ a *Ghimj(index,718);
        W_134 = W_134+ a *Ghimj(index,719);
        W_135 = W_135+ a *Ghimj(index,720);
        W_138 = W_138+ a *Ghimj(index,721);
        a = - W_118/ Ghimj(index,745);
        W_118 = -a;
        W_123 = W_123+ a *Ghimj(index,746);
        W_125 = W_125+ a *Ghimj(index,747);
        W_126 = W_126+ a *Ghimj(index,748);
        W_127 = W_127+ a *Ghimj(index,749);
        W_128 = W_128+ a *Ghimj(index,750);
        W_129 = W_129+ a *Ghimj(index,751);
        W_131 = W_131+ a *Ghimj(index,752);
        W_132 = W_132+ a *Ghimj(index,753);
        W_134 = W_134+ a *Ghimj(index,754);
        W_135 = W_135+ a *Ghimj(index,755);
        W_137 = W_137+ a *Ghimj(index,756);
        W_138 = W_138+ a *Ghimj(index,757);
        a = - W_119/ Ghimj(index,767);
        W_119 = -a;
        W_121 = W_121+ a *Ghimj(index,768);
        W_124 = W_124+ a *Ghimj(index,769);
        W_125 = W_125+ a *Ghimj(index,770);
        W_126 = W_126+ a *Ghimj(index,771);
        W_127 = W_127+ a *Ghimj(index,772);
        W_129 = W_129+ a *Ghimj(index,773);
        W_133 = W_133+ a *Ghimj(index,774);
        W_136 = W_136+ a *Ghimj(index,775);
        W_137 = W_137+ a *Ghimj(index,776);
        a = - W_120/ Ghimj(index,787);
        W_120 = -a;
        W_122 = W_122+ a *Ghimj(index,788);
        W_124 = W_124+ a *Ghimj(index,789);
        W_126 = W_126+ a *Ghimj(index,790);
        W_127 = W_127+ a *Ghimj(index,791);
        W_128 = W_128+ a *Ghimj(index,792);
        W_130 = W_130+ a *Ghimj(index,793);
        W_133 = W_133+ a *Ghimj(index,794);
        W_135 = W_135+ a *Ghimj(index,795);
        W_136 = W_136+ a *Ghimj(index,796);
        W_137 = W_137+ a *Ghimj(index,797);
        a = - W_121/ Ghimj(index,821);
        W_121 = -a;
        W_124 = W_124+ a *Ghimj(index,822);
        W_125 = W_125+ a *Ghimj(index,823);
        W_126 = W_126+ a *Ghimj(index,824);
        W_127 = W_127+ a *Ghimj(index,825);
        W_129 = W_129+ a *Ghimj(index,826);
        W_133 = W_133+ a *Ghimj(index,827);
        W_135 = W_135+ a *Ghimj(index,828);
        W_136 = W_136+ a *Ghimj(index,829);
        W_137 = W_137+ a *Ghimj(index,830);
        a = - W_122/ Ghimj(index,847);
        W_122 = -a;
        W_124 = W_124+ a *Ghimj(index,848);
        W_125 = W_125+ a *Ghimj(index,849);
        W_126 = W_126+ a *Ghimj(index,850);
        W_127 = W_127+ a *Ghimj(index,851);
        W_128 = W_128+ a *Ghimj(index,852);
        W_129 = W_129+ a *Ghimj(index,853);
        W_130 = W_130+ a *Ghimj(index,854);
        W_131 = W_131+ a *Ghimj(index,855);
        W_133 = W_133+ a *Ghimj(index,856);
        W_135 = W_135+ a *Ghimj(index,857);
        W_136 = W_136+ a *Ghimj(index,858);
        W_137 = W_137+ a *Ghimj(index,859);
        W_138 = W_138+ a *Ghimj(index,860);
        a = - W_123/ Ghimj(index,869);
        W_123 = -a;
        W_124 = W_124+ a *Ghimj(index,870);
        W_125 = W_125+ a *Ghimj(index,871);
        W_126 = W_126+ a *Ghimj(index,872);
        W_127 = W_127+ a *Ghimj(index,873);
        W_128 = W_128+ a *Ghimj(index,874);
        W_129 = W_129+ a *Ghimj(index,875);
        W_130 = W_130+ a *Ghimj(index,876);
        W_131 = W_131+ a *Ghimj(index,877);
        W_132 = W_132+ a *Ghimj(index,878);
        W_133 = W_133+ a *Ghimj(index,879);
        W_134 = W_134+ a *Ghimj(index,880);
        W_135 = W_135+ a *Ghimj(index,881);
        W_136 = W_136+ a *Ghimj(index,882);
        W_137 = W_137+ a *Ghimj(index,883);
        W_138 = W_138+ a *Ghimj(index,884);
        a = - W_124/ Ghimj(index,896);
        W_124 = -a;
        W_125 = W_125+ a *Ghimj(index,897);
        W_126 = W_126+ a *Ghimj(index,898);
        W_127 = W_127+ a *Ghimj(index,899);
        W_128 = W_128+ a *Ghimj(index,900);
        W_129 = W_129+ a *Ghimj(index,901);
        W_130 = W_130+ a *Ghimj(index,902);
        W_131 = W_131+ a *Ghimj(index,903);
        W_132 = W_132+ a *Ghimj(index,904);
        W_133 = W_133+ a *Ghimj(index,905);
        W_135 = W_135+ a *Ghimj(index,906);
        W_136 = W_136+ a *Ghimj(index,907);
        W_137 = W_137+ a *Ghimj(index,908);
        W_138 = W_138+ a *Ghimj(index,909);
        a = - W_125/ Ghimj(index,934);
        W_125 = -a;
        W_126 = W_126+ a *Ghimj(index,935);
        W_127 = W_127+ a *Ghimj(index,936);
        W_128 = W_128+ a *Ghimj(index,937);
        W_129 = W_129+ a *Ghimj(index,938);
        W_130 = W_130+ a *Ghimj(index,939);
        W_131 = W_131+ a *Ghimj(index,940);
        W_132 = W_132+ a *Ghimj(index,941);
        W_133 = W_133+ a *Ghimj(index,942);
        W_134 = W_134+ a *Ghimj(index,943);
        W_135 = W_135+ a *Ghimj(index,944);
        W_136 = W_136+ a *Ghimj(index,945);
        W_137 = W_137+ a *Ghimj(index,946);
        W_138 = W_138+ a *Ghimj(index,947);
        a = - W_126/ Ghimj(index,1023);
        W_126 = -a;
        W_127 = W_127+ a *Ghimj(index,1024);
        W_128 = W_128+ a *Ghimj(index,1025);
        W_129 = W_129+ a *Ghimj(index,1026);
        W_130 = W_130+ a *Ghimj(index,1027);
        W_131 = W_131+ a *Ghimj(index,1028);
        W_132 = W_132+ a *Ghimj(index,1029);
        W_133 = W_133+ a *Ghimj(index,1030);
        W_134 = W_134+ a *Ghimj(index,1031);
        W_135 = W_135+ a *Ghimj(index,1032);
        W_136 = W_136+ a *Ghimj(index,1033);
        W_137 = W_137+ a *Ghimj(index,1034);
        W_138 = W_138+ a *Ghimj(index,1035);
        a = - W_127/ Ghimj(index,1071);
        W_127 = -a;
        W_128 = W_128+ a *Ghimj(index,1072);
        W_129 = W_129+ a *Ghimj(index,1073);
        W_130 = W_130+ a *Ghimj(index,1074);
        W_131 = W_131+ a *Ghimj(index,1075);
        W_132 = W_132+ a *Ghimj(index,1076);
        W_133 = W_133+ a *Ghimj(index,1077);
        W_134 = W_134+ a *Ghimj(index,1078);
        W_135 = W_135+ a *Ghimj(index,1079);
        W_136 = W_136+ a *Ghimj(index,1080);
        W_137 = W_137+ a *Ghimj(index,1081);
        W_138 = W_138+ a *Ghimj(index,1082);
        a = - W_128/ Ghimj(index,1138);
        W_128 = -a;
        W_129 = W_129+ a *Ghimj(index,1139);
        W_130 = W_130+ a *Ghimj(index,1140);
        W_131 = W_131+ a *Ghimj(index,1141);
        W_132 = W_132+ a *Ghimj(index,1142);
        W_133 = W_133+ a *Ghimj(index,1143);
        W_134 = W_134+ a *Ghimj(index,1144);
        W_135 = W_135+ a *Ghimj(index,1145);
        W_136 = W_136+ a *Ghimj(index,1146);
        W_137 = W_137+ a *Ghimj(index,1147);
        W_138 = W_138+ a *Ghimj(index,1148);
        a = - W_129/ Ghimj(index,1176);
        W_129 = -a;
        W_130 = W_130+ a *Ghimj(index,1177);
        W_131 = W_131+ a *Ghimj(index,1178);
        W_132 = W_132+ a *Ghimj(index,1179);
        W_133 = W_133+ a *Ghimj(index,1180);
        W_134 = W_134+ a *Ghimj(index,1181);
        W_135 = W_135+ a *Ghimj(index,1182);
        W_136 = W_136+ a *Ghimj(index,1183);
        W_137 = W_137+ a *Ghimj(index,1184);
        W_138 = W_138+ a *Ghimj(index,1185);
        a = - W_130/ Ghimj(index,1218);
        W_130 = -a;
        W_131 = W_131+ a *Ghimj(index,1219);
        W_132 = W_132+ a *Ghimj(index,1220);
        W_133 = W_133+ a *Ghimj(index,1221);
        W_134 = W_134+ a *Ghimj(index,1222);
        W_135 = W_135+ a *Ghimj(index,1223);
        W_136 = W_136+ a *Ghimj(index,1224);
        W_137 = W_137+ a *Ghimj(index,1225);
        W_138 = W_138+ a *Ghimj(index,1226);
        a = - W_131/ Ghimj(index,1242);
        W_131 = -a;
        W_132 = W_132+ a *Ghimj(index,1243);
        W_133 = W_133+ a *Ghimj(index,1244);
        W_134 = W_134+ a *Ghimj(index,1245);
        W_135 = W_135+ a *Ghimj(index,1246);
        W_136 = W_136+ a *Ghimj(index,1247);
        W_137 = W_137+ a *Ghimj(index,1248);
        W_138 = W_138+ a *Ghimj(index,1249);
        a = - W_132/ Ghimj(index,1262);
        W_132 = -a;
        W_133 = W_133+ a *Ghimj(index,1263);
        W_134 = W_134+ a *Ghimj(index,1264);
        W_135 = W_135+ a *Ghimj(index,1265);
        W_136 = W_136+ a *Ghimj(index,1266);
        W_137 = W_137+ a *Ghimj(index,1267);
        W_138 = W_138+ a *Ghimj(index,1268);
        a = - W_133/ Ghimj(index,1297);
        W_133 = -a;
        W_134 = W_134+ a *Ghimj(index,1298);
        W_135 = W_135+ a *Ghimj(index,1299);
        W_136 = W_136+ a *Ghimj(index,1300);
        W_137 = W_137+ a *Ghimj(index,1301);
        W_138 = W_138+ a *Ghimj(index,1302);
        a = - W_134/ Ghimj(index,1324);
        W_134 = -a;
        W_135 = W_135+ a *Ghimj(index,1325);
        W_136 = W_136+ a *Ghimj(index,1326);
        W_137 = W_137+ a *Ghimj(index,1327);
        W_138 = W_138+ a *Ghimj(index,1328);
        a = - W_135/ Ghimj(index,1370);
        W_135 = -a;
        W_136 = W_136+ a *Ghimj(index,1371);
        W_137 = W_137+ a *Ghimj(index,1372);
        W_138 = W_138+ a *Ghimj(index,1373);
        a = - W_136/ Ghimj(index,1398);
        W_136 = -a;
        W_137 = W_137+ a *Ghimj(index,1399);
        W_138 = W_138+ a *Ghimj(index,1400);
        a = - W_137/ Ghimj(index,1451);
        W_137 = -a;
        W_138 = W_138+ a *Ghimj(index,1452);
        Ghimj(index,1453) = W_83;
        Ghimj(index,1454) = W_88;
        Ghimj(index,1455) = W_97;
        Ghimj(index,1456) = W_98;
        Ghimj(index,1457) = W_103;
        Ghimj(index,1458) = W_104;
        Ghimj(index,1459) = W_105;
        Ghimj(index,1460) = W_106;
        Ghimj(index,1461) = W_107;
        Ghimj(index,1462) = W_112;
        Ghimj(index,1463) = W_114;
        Ghimj(index,1464) = W_116;
        Ghimj(index,1465) = W_118;
        Ghimj(index,1466) = W_119;
        Ghimj(index,1467) = W_120;
        Ghimj(index,1468) = W_121;
        Ghimj(index,1469) = W_122;
        Ghimj(index,1470) = W_123;
        Ghimj(index,1471) = W_124;
        Ghimj(index,1472) = W_125;
        Ghimj(index,1473) = W_126;
        Ghimj(index,1474) = W_127;
        Ghimj(index,1475) = W_128;
        Ghimj(index,1476) = W_129;
        Ghimj(index,1477) = W_130;
        Ghimj(index,1478) = W_131;
        Ghimj(index,1479) = W_132;
        Ghimj(index,1480) = W_133;
        Ghimj(index,1481) = W_134;
        Ghimj(index,1482) = W_135;
        Ghimj(index,1483) = W_136;
        Ghimj(index,1484) = W_137;
        Ghimj(index,1485) = W_138;
}

__device__ void ros_Decomp(double * __restrict__ Ghimj, int &Ndec, int VL_GLO)
{
    kppDecomp(Ghimj, VL_GLO);
    Ndec++;
}


__device__ void ros_PrepareMatrix(double &H, int direction, double gam, double *jac0, double *Ghimj,  int &Nsng, int &Ndec, int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int ising, nConsecutive;
    double ghinv;
    

        ghinv = ONE/(direction*H*gam);
        for (int i=0; i<LU_NONZERO; i++)
	  Ghimj(index,i) = -jac0(index,i);

        Ghimj(index,0) += ghinv;
        Ghimj(index,1) += ghinv;
        Ghimj(index,2) += ghinv;
        Ghimj(index,3) += ghinv;
        Ghimj(index,4) += ghinv;
        Ghimj(index,5) += ghinv;
        Ghimj(index,6) += ghinv;
        Ghimj(index,9) += ghinv;
        Ghimj(index,25) += ghinv;
        Ghimj(index,29) += ghinv;
        Ghimj(index,38) += ghinv;
        Ghimj(index,43) += ghinv;
        Ghimj(index,46) += ghinv;
        Ghimj(index,48) += ghinv;
        Ghimj(index,52) += ghinv;
        Ghimj(index,58) += ghinv;
        Ghimj(index,60) += ghinv;
        Ghimj(index,62) += ghinv;
        Ghimj(index,64) += ghinv;
        Ghimj(index,68) += ghinv;
        Ghimj(index,69) += ghinv;
        Ghimj(index,72) += ghinv;
        Ghimj(index,75) += ghinv;
        Ghimj(index,112) += ghinv;
        Ghimj(index,123) += ghinv;
        Ghimj(index,140) += ghinv;
        Ghimj(index,148) += ghinv;
        Ghimj(index,163) += ghinv;
        Ghimj(index,170) += ghinv;
        Ghimj(index,182) += ghinv;
        Ghimj(index,185) += ghinv;
        Ghimj(index,190) += ghinv;
        Ghimj(index,194) += ghinv;
        Ghimj(index,202) += ghinv;
        Ghimj(index,206) += ghinv;
        Ghimj(index,233) += ghinv;
        Ghimj(index,244) += ghinv;
        Ghimj(index,251) += ghinv;
        Ghimj(index,255) += ghinv;
        Ghimj(index,258) += ghinv;
        Ghimj(index,260) += ghinv;
        Ghimj(index,262) += ghinv;
        Ghimj(index,264) += ghinv;
        Ghimj(index,266) += ghinv;
        Ghimj(index,268) += ghinv;
        Ghimj(index,270) += ghinv;
        Ghimj(index,272) += ghinv;
        Ghimj(index,276) += ghinv;
        Ghimj(index,278) += ghinv;
        Ghimj(index,280) += ghinv;
        Ghimj(index,282) += ghinv;
        Ghimj(index,285) += ghinv;
        Ghimj(index,288) += ghinv;
        Ghimj(index,290) += ghinv;
        Ghimj(index,292) += ghinv;
        Ghimj(index,294) += ghinv;
        Ghimj(index,296) += ghinv;
        Ghimj(index,300) += ghinv;
        Ghimj(index,303) += ghinv;
        Ghimj(index,306) += ghinv;
        Ghimj(index,310) += ghinv;
        Ghimj(index,315) += ghinv;
        Ghimj(index,319) += ghinv;
        Ghimj(index,323) += ghinv;
        Ghimj(index,327) += ghinv;
        Ghimj(index,331) += ghinv;
        Ghimj(index,335) += ghinv;
        Ghimj(index,339) += ghinv;
        Ghimj(index,343) += ghinv;
        Ghimj(index,347) += ghinv;
        Ghimj(index,352) += ghinv;
        Ghimj(index,356) += ghinv;
        Ghimj(index,360) += ghinv;
        Ghimj(index,364) += ghinv;
        Ghimj(index,368) += ghinv;
        Ghimj(index,374) += ghinv;
        Ghimj(index,377) += ghinv;
        Ghimj(index,382) += ghinv;
        Ghimj(index,386) += ghinv;
        Ghimj(index,393) += ghinv;
        Ghimj(index,397) += ghinv;
        Ghimj(index,405) += ghinv;
        Ghimj(index,412) += ghinv;
        Ghimj(index,416) += ghinv;
        Ghimj(index,421) += ghinv;
        Ghimj(index,427) += ghinv;
        Ghimj(index,436) += ghinv;
        Ghimj(index,444) += ghinv;
        Ghimj(index,450) += ghinv;
        Ghimj(index,457) += ghinv;
        Ghimj(index,469) += ghinv;
        Ghimj(index,481) += ghinv;
        Ghimj(index,489) += ghinv;
        Ghimj(index,497) += ghinv;
        Ghimj(index,505) += ghinv;
        Ghimj(index,514) += ghinv;
        Ghimj(index,538) += ghinv;
        Ghimj(index,549) += ghinv;
        Ghimj(index,557) += ghinv;
        Ghimj(index,565) += ghinv;
        Ghimj(index,573) += ghinv;
        Ghimj(index,586) += ghinv;
        Ghimj(index,600) += ghinv;
        Ghimj(index,605) += ghinv;
        Ghimj(index,610) += ghinv;
        Ghimj(index,616) += ghinv;
        Ghimj(index,622) += ghinv;
        Ghimj(index,626) += ghinv;
        Ghimj(index,636) += ghinv;
        Ghimj(index,648) += ghinv;
        Ghimj(index,659) += ghinv;
        Ghimj(index,669) += ghinv;
        Ghimj(index,677) += ghinv;
        Ghimj(index,689) += ghinv;
        Ghimj(index,697) += ghinv;
        Ghimj(index,706) += ghinv;
        Ghimj(index,714) += ghinv;
        Ghimj(index,731) += ghinv;
        Ghimj(index,745) += ghinv;
        Ghimj(index,767) += ghinv;
        Ghimj(index,787) += ghinv;
        Ghimj(index,821) += ghinv;
        Ghimj(index,847) += ghinv;
        Ghimj(index,869) += ghinv;
        Ghimj(index,896) += ghinv;
        Ghimj(index,934) += ghinv;
        Ghimj(index,1023) += ghinv;
        Ghimj(index,1071) += ghinv;
        Ghimj(index,1138) += ghinv;
        Ghimj(index,1176) += ghinv;
        Ghimj(index,1218) += ghinv;
        Ghimj(index,1242) += ghinv;
        Ghimj(index,1262) += ghinv;
        Ghimj(index,1297) += ghinv;
        Ghimj(index,1324) += ghinv;
        Ghimj(index,1370) += ghinv;
        Ghimj(index,1398) += ghinv;
        Ghimj(index,1451) += ghinv;
        Ghimj(index,1485) += ghinv;
        Ghimj(index,1486) += ghinv;
        ros_Decomp(Ghimj, Ndec, VL_GLO);
}

__device__ void Jac_sp(const double * __restrict__ var, const double * __restrict__ fix,
                 const double * __restrict__ rconst, double * __restrict__ jcb, int &Njac, const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

 double dummy, B_0, B_1, B_2, B_3, B_4, B_5, B_6, B_7, B_8, B_9, B_10, B_11, B_12, B_13, B_14, B_15, B_16, B_17, B_18, B_19, B_20, B_21, B_22, B_23, B_24, B_25, B_26, B_27, B_28, B_29, B_30, B_31, B_32, B_33, B_34, B_35, B_36, B_37, B_38, B_39, B_40, B_41, B_42, B_43, B_44, B_45, B_46, B_47, B_48, B_49, B_50, B_51, B_52, B_53, B_54, B_55, B_56, B_57, B_58, B_59, B_60, B_61, B_62, B_63, B_64, B_65, B_66, B_67, B_68, B_69, B_70, B_71, B_72, B_73, B_74, B_75, B_76, B_77, B_78, B_79, B_80, B_81, B_82, B_83, B_84, B_85, B_86, B_87, B_88, B_89, B_90, B_91, B_92, B_93, B_94, B_95, B_96, B_97, B_98, B_99, B_100, B_101, B_102, B_103, B_104, B_105, B_106, B_107, B_108, B_109, B_110, B_111, B_112, B_113, B_114, B_115, B_116, B_117, B_118, B_119, B_120, B_121, B_122, B_123, B_124, B_125, B_126, B_127, B_128, B_129, B_130, B_131, B_132, B_133, B_134, B_135, B_136, B_137, B_138, B_139, B_140, B_141, B_142, B_143, B_144, B_145, B_146, B_147, B_148, B_149, B_150, B_151, B_152, B_153, B_154, B_155, B_156, B_157, B_158, B_159, B_160, B_161, B_162, B_163, B_164, B_165, B_166, B_167, B_168, B_169, B_170, B_171, B_172, B_173, B_174, B_175, B_176, B_177, B_178, B_179, B_180, B_181, B_182, B_183, B_184, B_185, B_186, B_187, B_188, B_189, B_190, B_191, B_192, B_193, B_194, B_195, B_196, B_197, B_198, B_199, B_200, B_201, B_202, B_203, B_204, B_205, B_206, B_207, B_208, B_209, B_210, B_211, B_212, B_213, B_214, B_215, B_216, B_217, B_218, B_219, B_220, B_221, B_222, B_223, B_224, B_225, B_226, B_227, B_228, B_229, B_230, B_231, B_232, B_233, B_234, B_235, B_236, B_237, B_238, B_239, B_240, B_241, B_242, B_243, B_244, B_245, B_246, B_247, B_248, B_249, B_250, B_251, B_252, B_253, B_254, B_255, B_256, B_257, B_258, B_259, B_260, B_261, B_262, B_263, B_264, B_265, B_266, B_267, B_268, B_269, B_270, B_271, B_272, B_273, B_274, B_275, B_276, B_277, B_278, B_279, B_280, B_281, B_282, B_283, B_284, B_285, B_286, B_287, B_288, B_289, B_290, B_291, B_292, B_293, B_294, B_295, B_296, B_297, B_298, B_299, B_300, B_301, B_302, B_303, B_304, B_305, B_306, B_307, B_308, B_309, B_310, B_311, B_312, B_313, B_314, B_315, B_316, B_317, B_318, B_319, B_320, B_321, B_322, B_323, B_324, B_325, B_326, B_327, B_328, B_329, B_330, B_331, B_332, B_333, B_334, B_335, B_336, B_337, B_338, B_339, B_340, B_341, B_342, B_343, B_344, B_345, B_346, B_347, B_348, B_349, B_350, B_351, B_352, B_353, B_354, B_355, B_356, B_357, B_358, B_359, B_360, B_361, B_362, B_363, B_364, B_365, B_366, B_367, B_368, B_369, B_370, B_371, B_372, B_373, B_374, B_375, B_376, B_377, B_378, B_379, B_380, B_381, B_382, B_383, B_384, B_385, B_386, B_387, B_388, B_389, B_390, B_391, B_392, B_393, B_394, B_395, B_396, B_397, B_398, B_399, B_400, B_401, B_402, B_403, B_404, B_405, B_406, B_407, B_408, B_409, B_410, B_411, B_412, B_413, B_414, B_415, B_416, B_417, B_418, B_419, B_420, B_421, B_422, B_423, B_424, B_425, B_426, B_427, B_428, B_429, B_430, B_431, B_432, B_433, B_434, B_435, B_436, B_437, B_438, B_439, B_440, B_441, B_442, B_443, B_444, B_445, B_446, B_447, B_448, B_449, B_450, B_451, B_452, B_453, B_454, B_455, B_456, B_457, B_458, B_459, B_460, B_461, B_462, B_463, B_464, B_465, B_466, B_467, B_468, B_469, B_470, B_471, B_472, B_473, B_474, B_475, B_476, B_477, B_478, B_479, B_480, B_481, B_482, B_483, B_484, B_485, B_486, B_487, B_488, B_489, B_490, B_491, B_492, B_493, B_494, B_495, B_496, B_497, B_498, B_499, B_500, B_501, B_502, B_503, B_504, B_505, B_506, B_507, B_508, B_509, B_510, B_511, B_512, B_513, B_514, B_515, B_516, B_517, B_518, B_519, B_520, B_521, B_522;


    Njac++;

    B_0 = rconst(index,0)*fix(index,0);
        B_2 = rconst(index,1)*fix(index,0);
        B_4 = 1.2e-10*var(index,124);
        B_5 = 1.2e-10*var(index,120);
        B_6 = rconst(index,3)*var(index,131);
        B_7 = rconst(index,3)*var(index,124);
        B_8 = rconst(index,4)*fix(index,0);
        B_10 = rconst(index,5)*var(index,124);
        B_11 = rconst(index,5)*var(index,122);
        B_12 = 1.2e-10*var(index,120);
        B_13 = 1.2e-10*var(index,97);
        B_14 = rconst(index,7)*var(index,131);
        B_15 = rconst(index,7)*var(index,126);
        B_16 = rconst(index,8)*var(index,126);
        B_17 = rconst(index,8)*var(index,124);
        B_18 = rconst(index,9)*var(index,126);
        B_19 = rconst(index,9)*var(index,97);
        B_20 = rconst(index,10)*var(index,137);
        B_21 = rconst(index,10)*var(index,131);
        B_22 = rconst(index,11)*var(index,137);
        B_23 = rconst(index,11)*var(index,124);
        B_24 = 7.2e-11*var(index,137);
        B_25 = 7.2e-11*var(index,122);
        B_26 = 6.9e-12*var(index,137);
        B_27 = 6.9e-12*var(index,122);
        B_28 = 1.6e-12*var(index,137);
        B_29 = 1.6e-12*var(index,122);
        B_30 = rconst(index,15)*var(index,137);
        B_31 = rconst(index,15)*var(index,126);
        B_32 = rconst(index,16)*2*var(index,137);
        B_33 = rconst(index,17)*var(index,128);
        B_34 = rconst(index,17)*var(index,120);
        B_35 = 1.8e-12*var(index,126);
        B_36 = 1.8e-12*var(index,88);
        B_37 = rconst(index,19)*fix(index,0);
        B_39 = rconst(index,20)*fix(index,1);
        B_41 = rconst(index,21)*var(index,120);
        B_42 = rconst(index,21)*var(index,60);
        B_43 = rconst(index,22)*var(index,120);
        B_44 = rconst(index,22)*var(index,60);
        B_45 = rconst(index,23)*var(index,133);
        B_46 = rconst(index,23)*var(index,124);
        B_47 = rconst(index,24)*var(index,133);
        B_48 = rconst(index,24)*var(index,59);
        B_49 = rconst(index,25)*var(index,135);
        B_50 = rconst(index,25)*var(index,131);
        B_51 = rconst(index,26)*var(index,135);
        B_52 = rconst(index,26)*var(index,124);
        B_53 = rconst(index,27)*var(index,135);
        B_54 = rconst(index,27)*var(index,59);
        B_55 = rconst(index,28)*var(index,136);
        B_56 = rconst(index,28)*var(index,133);
        B_57 = rconst(index,29)*var(index,136);
        B_58 = rconst(index,29)*var(index,135);
        B_59 = rconst(index,30);
        B_60 = rconst(index,31)*var(index,133);
        B_61 = rconst(index,31)*var(index,126);
        B_62 = rconst(index,32)*var(index,137);
        B_63 = rconst(index,32)*var(index,133);
        B_64 = rconst(index,33)*var(index,135);
        B_65 = rconst(index,33)*var(index,126);
        B_66 = rconst(index,34)*var(index,137);
        B_67 = rconst(index,34)*var(index,135);
        B_68 = 3.5e-12*var(index,137);
        B_69 = 3.5e-12*var(index,136);
        B_70 = rconst(index,36)*var(index,126);
        B_71 = rconst(index,36)*var(index,76);
        B_72 = rconst(index,37)*var(index,126);
        B_73 = rconst(index,37)*var(index,101);
        B_74 = rconst(index,38);
        B_75 = rconst(index,39)*var(index,126);
        B_76 = rconst(index,39)*var(index,73);
        B_77 = rconst(index,40)*var(index,126);
        B_78 = rconst(index,40)*var(index,47);
        B_79 = rconst(index,41)*var(index,124);
        B_80 = rconst(index,41)*var(index,92);
        B_81 = rconst(index,42)*var(index,137);
        B_82 = rconst(index,42)*var(index,92);
        B_83 = rconst(index,43)*var(index,137);
        B_84 = rconst(index,43)*var(index,92);
        B_85 = rconst(index,44)*var(index,133);
        B_86 = rconst(index,44)*var(index,92);
        B_87 = rconst(index,45)*var(index,133);
        B_88 = rconst(index,45)*var(index,92);
        B_89 = rconst(index,46)*var(index,135);
        B_90 = rconst(index,46)*var(index,92);
        B_91 = rconst(index,47)*var(index,135);
        B_92 = rconst(index,47)*var(index,92);
        B_93 = 1.2e-14*var(index,124);
        B_94 = 1.2e-14*var(index,84);
        B_95 = 1300;
        B_96 = rconst(index,50)*var(index,126);
        B_97 = rconst(index,50)*var(index,87);
        B_98 = rconst(index,51)*var(index,87);
        B_99 = rconst(index,51)*var(index,70);
        B_100 = rconst(index,52)*var(index,135);
        B_101 = rconst(index,52)*var(index,87);
        B_102 = 1.66e-12*var(index,126);
        B_103 = 1.66e-12*var(index,70);
        B_104 = rconst(index,54)*var(index,126);
        B_105 = rconst(index,54)*var(index,61);
        B_106 = rconst(index,55)*fix(index,0);
        B_108 = 1.75e-10*var(index,120);
        B_109 = 1.75e-10*var(index,98);
        B_110 = rconst(index,57)*var(index,126);
        B_111 = rconst(index,57)*var(index,98);
        B_112 = rconst(index,58)*var(index,126);
        B_113 = rconst(index,58)*var(index,89);
        B_114 = rconst(index,59)*var(index,137);
        B_115 = rconst(index,59)*var(index,125);
        B_116 = rconst(index,60)*var(index,133);
        B_117 = rconst(index,60)*var(index,125);
        B_118 = 1.3e-12*var(index,136);
        B_119 = 1.3e-12*var(index,125);
        B_120 = rconst(index,62)*2*var(index,125);
        B_121 = rconst(index,63)*2*var(index,125);
        B_122 = rconst(index,64)*var(index,126);
        B_123 = rconst(index,64)*var(index,104);
        B_124 = rconst(index,65)*var(index,130);
        B_125 = rconst(index,65)*var(index,126);
        B_126 = rconst(index,66)*var(index,136);
        B_127 = rconst(index,66)*var(index,130);
        B_128 = rconst(index,67)*var(index,126);
        B_129 = rconst(index,67)*var(index,95);
        B_130 = 4e-13*var(index,126);
        B_131 = 4e-13*var(index,78);
        B_132 = rconst(index,69)*var(index,126);
        B_133 = rconst(index,69)*var(index,48);
        B_134 = rconst(index,70)*var(index,124);
        B_135 = rconst(index,70)*var(index,103);
        B_136 = rconst(index,71)*var(index,126);
        B_137 = rconst(index,71)*var(index,103);
        B_138 = rconst(index,72)*var(index,137);
        B_139 = rconst(index,72)*var(index,117);
        B_140 = rconst(index,73)*var(index,133);
        B_141 = rconst(index,73)*var(index,117);
        B_142 = 2.3e-12*var(index,136);
        B_143 = 2.3e-12*var(index,117);
        B_144 = rconst(index,75)*var(index,125);
        B_145 = rconst(index,75)*var(index,117);
        B_146 = rconst(index,76)*var(index,126);
        B_147 = rconst(index,76)*var(index,71);
        B_148 = rconst(index,77)*var(index,126);
        B_149 = rconst(index,77)*var(index,119);
        B_150 = rconst(index,78)*var(index,136);
        B_151 = rconst(index,78)*var(index,119);
        B_152 = rconst(index,79)*var(index,126);
        B_153 = rconst(index,79)*var(index,74);
        B_154 = rconst(index,80)*var(index,137);
        B_155 = rconst(index,80)*var(index,121);
        B_156 = rconst(index,81)*var(index,137);
        B_157 = rconst(index,81)*var(index,121);
        B_158 = rconst(index,82)*var(index,133);
        B_159 = rconst(index,82)*var(index,121);
        B_160 = rconst(index,83)*var(index,135);
        B_161 = rconst(index,83)*var(index,121);
        B_162 = 4e-12*var(index,136);
        B_163 = 4e-12*var(index,121);
        B_164 = rconst(index,85)*var(index,125);
        B_165 = rconst(index,85)*var(index,121);
        B_166 = rconst(index,86)*var(index,125);
        B_167 = rconst(index,86)*var(index,121);
        B_168 = rconst(index,87)*var(index,121);
        B_169 = rconst(index,87)*var(index,117);
        B_170 = rconst(index,88)*2*var(index,121);
        B_171 = rconst(index,89)*var(index,126);
        B_172 = rconst(index,89)*var(index,63);
        B_173 = rconst(index,90)*var(index,126);
        B_174 = rconst(index,90)*var(index,58);
        B_175 = rconst(index,91)*var(index,126);
        B_176 = rconst(index,91)*var(index,77);
        B_177 = rconst(index,92);
        B_178 = rconst(index,93)*var(index,126);
        B_179 = rconst(index,93)*var(index,49);
        B_180 = rconst(index,94)*var(index,124);
        B_181 = rconst(index,94)*var(index,107);
        B_182 = rconst(index,95)*var(index,126);
        B_183 = rconst(index,95)*var(index,107);
        B_184 = rconst(index,96)*var(index,136);
        B_185 = rconst(index,96)*var(index,107);
        B_186 = rconst(index,97)*var(index,137);
        B_187 = rconst(index,97)*var(index,93);
        B_188 = rconst(index,98)*var(index,133);
        B_189 = rconst(index,98)*var(index,93);
        B_190 = rconst(index,99)*var(index,125);
        B_191 = rconst(index,99)*var(index,93);
        B_192 = rconst(index,100)*var(index,126);
        B_193 = rconst(index,100)*var(index,69);
        B_194 = rconst(index,101)*var(index,137);
        B_195 = rconst(index,101)*var(index,115);
        B_196 = rconst(index,102)*var(index,133);
        B_197 = rconst(index,102)*var(index,115);
        B_198 = rconst(index,103)*var(index,126);
        B_199 = rconst(index,103)*var(index,67);
        B_200 = rconst(index,104)*var(index,126);
        B_201 = rconst(index,104)*var(index,86);
        B_202 = rconst(index,105)*var(index,137);
        B_203 = rconst(index,105)*var(index,94);
        B_204 = rconst(index,106)*var(index,133);
        B_205 = rconst(index,106)*var(index,94);
        B_206 = rconst(index,107)*var(index,125);
        B_207 = rconst(index,107)*var(index,94);
        B_208 = rconst(index,108)*var(index,126);
        B_209 = rconst(index,108)*var(index,72);
        B_210 = rconst(index,109)*var(index,126);
        B_211 = rconst(index,109)*var(index,108);
        B_212 = rconst(index,110)*var(index,126);
        B_213 = rconst(index,110)*var(index,96);
        B_214 = rconst(index,111)*var(index,126);
        B_215 = rconst(index,111)*var(index,62);
        B_216 = rconst(index,112)*var(index,126);
        B_217 = rconst(index,112)*var(index,40);
        B_218 = rconst(index,113)*var(index,125);
        B_219 = rconst(index,113)*var(index,102);
        B_220 = rconst(index,114)*var(index,137);
        B_221 = rconst(index,114)*var(index,102);
        B_222 = rconst(index,115)*var(index,133);
        B_223 = rconst(index,115)*var(index,102);
        B_224 = rconst(index,116)*var(index,126);
        B_225 = rconst(index,116)*var(index,79);
        B_226 = rconst(index,117)*var(index,124);
        B_227 = rconst(index,117)*var(index,110);
        B_228 = rconst(index,118)*var(index,126);
        B_229 = rconst(index,118)*var(index,110);
        B_230 = rconst(index,119)*var(index,137);
        B_231 = rconst(index,119)*var(index,113);
        B_232 = rconst(index,120)*var(index,133);
        B_233 = rconst(index,120)*var(index,113);
        B_234 = rconst(index,121)*var(index,135);
        B_235 = rconst(index,121)*var(index,113);
        B_236 = 2e-12*var(index,125);
        B_237 = 2e-12*var(index,113);
        B_238 = 2e-12*2*var(index,113);
        B_239 = 3e-11*var(index,126);
        B_240 = 3e-11*var(index,82);
        B_241 = rconst(index,125)*var(index,126);
        B_242 = rconst(index,125)*var(index,85);
        B_243 = rconst(index,126)*var(index,137);
        B_244 = rconst(index,126)*var(index,99);
        B_245 = rconst(index,127)*var(index,133);
        B_246 = rconst(index,127)*var(index,99);
        B_247 = rconst(index,128)*var(index,126);
        B_248 = rconst(index,128)*var(index,68);
        B_249 = 1.7e-12*var(index,126);
        B_250 = 1.7e-12*var(index,111);
        B_251 = 3.2e-11*var(index,126);
        B_252 = 3.2e-11*var(index,64);
        B_253 = rconst(index,131);
        B_254 = rconst(index,132)*var(index,124);
        B_255 = rconst(index,132)*var(index,106);
        B_256 = rconst(index,133)*var(index,126);
        B_257 = rconst(index,133)*var(index,106);
        B_258 = rconst(index,134)*var(index,136);
        B_259 = rconst(index,134)*var(index,106);
        B_260 = rconst(index,135)*var(index,137);
        B_261 = rconst(index,135)*var(index,109);
        B_262 = rconst(index,136)*var(index,133);
        B_263 = rconst(index,136)*var(index,109);
        B_264 = 2e-12*var(index,125);
        B_265 = 2e-12*var(index,109);
        B_266 = 2e-12*2*var(index,109);
        B_267 = 1e-10*var(index,126);
        B_268 = 1e-10*var(index,66);
        B_269 = 1.3e-11*var(index,126);
        B_270 = 1.3e-11*var(index,91);
        B_271 = rconst(index,141)*var(index,127);
        B_272 = rconst(index,141)*var(index,124);
        B_273 = rconst(index,142)*var(index,134);
        B_274 = rconst(index,142)*var(index,131);
        B_275 = rconst(index,143)*2*var(index,134);
        B_276 = rconst(index,144)*2*var(index,134);
        B_277 = rconst(index,145)*2*var(index,134);
        B_278 = rconst(index,146)*2*var(index,134);
        B_279 = rconst(index,147);
        B_280 = rconst(index,148)*var(index,127);
        B_281 = rconst(index,148)*var(index,97);
        B_282 = rconst(index,149)*var(index,137);
        B_283 = rconst(index,149)*var(index,127);
        B_284 = rconst(index,150)*var(index,137);
        B_285 = rconst(index,150)*var(index,127);
        B_286 = rconst(index,151)*var(index,127);
        B_287 = rconst(index,151)*var(index,88);
        B_288 = rconst(index,152)*var(index,134);
        B_289 = rconst(index,152)*var(index,126);
        B_290 = rconst(index,153)*var(index,137);
        B_291 = rconst(index,153)*var(index,134);
        B_292 = rconst(index,154)*var(index,138);
        B_293 = rconst(index,154)*var(index,126);
        B_294 = rconst(index,155)*var(index,126);
        B_295 = rconst(index,155)*var(index,112);
        B_296 = rconst(index,156)*var(index,134);
        B_297 = rconst(index,156)*var(index,133);
        B_298 = rconst(index,157)*var(index,135);
        B_299 = rconst(index,157)*var(index,134);
        B_300 = rconst(index,158);
        B_301 = rconst(index,159)*var(index,131);
        B_302 = rconst(index,159)*var(index,116);
        B_303 = rconst(index,160)*var(index,127);
        B_304 = rconst(index,160)*var(index,116);
        B_305 = rconst(index,161)*var(index,127);
        B_306 = rconst(index,161)*var(index,98);
        B_307 = rconst(index,162)*var(index,130);
        B_308 = rconst(index,162)*var(index,127);
        B_309 = 5.9e-11*var(index,127);
        B_310 = 5.9e-11*var(index,104);
        B_311 = rconst(index,164)*var(index,134);
        B_312 = rconst(index,164)*var(index,125);
        B_313 = 3.3e-10*var(index,120);
        B_314 = 3.3e-10*var(index,41);
        B_315 = 1.65e-10*var(index,120);
        B_316 = 1.65e-10*var(index,75);
        B_317 = rconst(index,167)*var(index,126);
        B_318 = rconst(index,167)*var(index,75);
        B_319 = 3.25e-10*var(index,120);
        B_320 = 3.25e-10*var(index,57);
        B_321 = rconst(index,169)*var(index,126);
        B_322 = rconst(index,169)*var(index,57);
        B_323 = rconst(index,170)*var(index,127);
        B_324 = rconst(index,170)*var(index,103);
        B_325 = 8e-11*var(index,127);
        B_326 = 8e-11*var(index,119);
        B_327 = 1.4e-10*var(index,120);
        B_328 = 1.4e-10*var(index,42);
        B_329 = 2.3e-10*var(index,120);
        B_330 = 2.3e-10*var(index,43);
        B_331 = rconst(index,174)*var(index,129);
        B_332 = rconst(index,174)*var(index,124);
        B_333 = rconst(index,175)*var(index,132);
        B_334 = rconst(index,175)*var(index,131);
        B_335 = 2.7e-12*2*var(index,132);
        B_336 = rconst(index,177)*2*var(index,132);
        B_337 = rconst(index,178)*var(index,137);
        B_338 = rconst(index,178)*var(index,129);
        B_339 = rconst(index,179)*var(index,137);
        B_340 = rconst(index,179)*var(index,132);
        B_341 = rconst(index,180)*var(index,126);
        B_342 = rconst(index,180)*var(index,123);
        B_343 = rconst(index,181)*var(index,131);
        B_344 = rconst(index,181)*var(index,118);
        B_345 = rconst(index,182)*var(index,126);
        B_346 = rconst(index,182)*var(index,100);
        B_347 = 4.9e-11*var(index,129);
        B_348 = 4.9e-11*var(index,105);
        B_349 = rconst(index,184)*var(index,133);
        B_350 = rconst(index,184)*var(index,132);
        B_351 = rconst(index,185)*var(index,135);
        B_352 = rconst(index,185)*var(index,132);
        B_353 = rconst(index,186);
        B_354 = rconst(index,187)*var(index,130);
        B_355 = rconst(index,187)*var(index,129);
        B_356 = rconst(index,188)*var(index,129);
        B_357 = rconst(index,188)*var(index,104);
        B_358 = rconst(index,189)*var(index,132);
        B_359 = rconst(index,189)*var(index,125);
        B_360 = rconst(index,190)*var(index,132);
        B_361 = rconst(index,190)*var(index,125);
        B_362 = rconst(index,191)*var(index,126);
        B_363 = rconst(index,191)*var(index,53);
        B_364 = rconst(index,192)*var(index,129);
        B_365 = rconst(index,192)*var(index,103);
        B_366 = rconst(index,193)*var(index,129);
        B_367 = rconst(index,193)*var(index,119);
        B_368 = rconst(index,194)*var(index,126);
        B_369 = rconst(index,194)*var(index,45);
        B_370 = rconst(index,195)*var(index,126);
        B_371 = rconst(index,195)*var(index,44);
        B_372 = 3.32e-15*var(index,129);
        B_373 = 3.32e-15*var(index,90);
        B_374 = 1.1e-15*var(index,129);
        B_375 = 1.1e-15*var(index,80);
        B_376 = rconst(index,198)*var(index,127);
        B_377 = rconst(index,198)*var(index,100);
        B_378 = rconst(index,199)*var(index,134);
        B_379 = rconst(index,199)*var(index,132);
        B_380 = rconst(index,200)*var(index,134);
        B_381 = rconst(index,200)*var(index,132);
        B_382 = rconst(index,201)*var(index,134);
        B_383 = rconst(index,201)*var(index,132);
        B_384 = 1.45e-11*var(index,127);
        B_385 = 1.45e-11*var(index,90);
        B_386 = rconst(index,203)*var(index,126);
        B_387 = rconst(index,203)*var(index,54);
        B_388 = rconst(index,204)*var(index,126);
        B_389 = rconst(index,204)*var(index,55);
        B_390 = rconst(index,205)*var(index,126);
        B_391 = rconst(index,205)*var(index,52);
        B_392 = rconst(index,206)*var(index,126);
        B_393 = rconst(index,206)*var(index,56);
        B_394 = rconst(index,207)*var(index,126);
        B_395 = rconst(index,207)*var(index,114);
        B_396 = rconst(index,208)*var(index,126);
        B_397 = rconst(index,208)*var(index,114);
        B_398 = rconst(index,209)*var(index,136);
        B_399 = rconst(index,209)*var(index,114);
        B_400 = 1e-10*var(index,126);
        B_401 = 1e-10*var(index,65);
        B_402 = rconst(index,211);
        B_403 = 3e-13*var(index,124);
        B_404 = 3e-13*var(index,81);
        B_405 = 5e-11*var(index,137);
        B_406 = 5e-11*var(index,46);
        B_407 = 3.3e-10*var(index,127);
        B_408 = 3.3e-10*var(index,114);
        B_409 = rconst(index,215)*var(index,129);
        B_410 = rconst(index,215)*var(index,114);
        B_411 = 4.4e-13*var(index,132);
        B_412 = 4.4e-13*var(index,114);
        B_414 = rconst(index,218);
        B_415 = rconst(index,219);
        B_416 = rconst(index,220);
        B_417 = rconst(index,221);
        B_418 = rconst(index,222);
        B_419 = rconst(index,223);
        B_420 = rconst(index,224);
        B_421 = rconst(index,225);
        B_422 = rconst(index,226);
        B_423 = rconst(index,227);
        B_424 = rconst(index,228);
        B_425 = rconst(index,229);
        B_426 = rconst(index,230);
        B_427 = rconst(index,231);
        B_428 = rconst(index,232);
        B_429 = rconst(index,233);
        B_431 = rconst(index,235);
        B_432 = rconst(index,236);
        B_433 = rconst(index,237);
        B_434 = rconst(index,238);
        B_435 = rconst(index,239);
        B_436 = rconst(index,240);
        B_437 = rconst(index,241);
        B_438 = rconst(index,242);
        B_439 = rconst(index,243);
        B_440 = rconst(index,244);
        B_441 = rconst(index,245);
        B_442 = rconst(index,246);
        B_443 = rconst(index,247);
        B_444 = rconst(index,248);
        B_445 = rconst(index,249);
        B_446 = rconst(index,250);
        B_447 = rconst(index,251);
        B_448 = rconst(index,252);
        B_449 = rconst(index,253);
        B_450 = rconst(index,254);
        B_451 = rconst(index,255);
        B_452 = rconst(index,256);
        B_453 = rconst(index,257);
        B_454 = rconst(index,258);
        B_455 = rconst(index,259);
        B_456 = rconst(index,260);
        B_457 = rconst(index,261);
        B_458 = rconst(index,262);
        B_459 = rconst(index,263);
        B_460 = rconst(index,264);
        B_461 = rconst(index,265);
        B_462 = rconst(index,266);
        B_463 = rconst(index,267);
        B_464 = rconst(index,268);
        B_465 = rconst(index,269);
        B_466 = rconst(index,270);
        B_467 = rconst(index,271);
        B_468 = rconst(index,272);
        B_469 = rconst(index,273);
        B_470 = rconst(index,274);
        B_471 = rconst(index,275);
        B_472 = rconst(index,276);
        B_473 = rconst(index,277);
        B_474 = rconst(index,278);
        B_475 = rconst(index,279);
        B_476 = rconst(index,280);
        B_477 = rconst(index,281);
        B_478 = rconst(index,282);
        B_479 = rconst(index,283);
        B_480 = rconst(index,284);
        B_481 = rconst(index,285)*var(index,128);
        B_482 = rconst(index,285)*var(index,83);
        B_483 = rconst(index,286);
        B_484 = rconst(index,287)*var(index,138);
        B_485 = rconst(index,287)*var(index,112);
        B_486 = rconst(index,288)*var(index,138);
        B_487 = rconst(index,288)*var(index,116);
        B_488 = rconst(index,289)*var(index,128);
        B_489 = rconst(index,289)*var(index,116);
        B_490 = rconst(index,290)*var(index,138);
        B_491 = rconst(index,290)*var(index,83);
        B_492 = rconst(index,291)*var(index,123);
        B_493 = rconst(index,291)*var(index,118);
        B_494 = rconst(index,292)*var(index,128);
        B_495 = rconst(index,292)*var(index,105);
        B_496 = rconst(index,293)*var(index,123);
        B_497 = rconst(index,293)*var(index,116);
        B_498 = rconst(index,294)*var(index,138);
        B_499 = rconst(index,294)*var(index,105);
        B_500 = rconst(index,295)*var(index,123);
        B_501 = rconst(index,295)*var(index,112);
        B_502 = rconst(index,296)*var(index,138);
        B_503 = rconst(index,296)*var(index,118);
        B_504 = rconst(index,297);
        B_505 = 2.3e-10*var(index,120);
        B_506 = 2.3e-10*var(index,15);
        B_507 = rconst(index,299);
        B_508 = 1.4e-10*var(index,120);
        B_509 = 1.4e-10*var(index,16);
        B_510 = rconst(index,301);
        B_511 = rconst(index,302)*var(index,120);
        B_512 = rconst(index,302)*var(index,17);
        B_513 = rconst(index,303)*var(index,120);
        B_514 = rconst(index,303)*var(index,17);
        B_515 = rconst(index,304);
        B_516 = 3e-10*var(index,120);
        B_517 = 3e-10*var(index,18);
        B_518 = rconst(index,306)*var(index,126);
        B_519 = rconst(index,306)*var(index,18);
        B_520 = rconst(index,307);
        B_521 = rconst(index,308);
        B_522 = rconst(index,309);
        jcb(index,0) = - B_469;
        jcb(index,1) = - B_476;
        jcb(index,2) = - B_474;
        jcb(index,3) = - B_480;
        jcb(index,4) = - B_504;
        jcb(index,5) = - B_521;
        jcb(index,6) = - B_522;
        jcb(index,7) = B_476;
        jcb(index,8) = B_474;
        jcb(index,9) = 0;
        jcb(index,10) = B_313+ B_462;
        jcb(index,11) = B_327+ B_465;
        jcb(index,12) = B_329+ B_464;
        jcb(index,13) = B_370+ B_472;
        jcb(index,14) = B_368+ B_473;
        jcb(index,15) = B_390+ B_477;
        jcb(index,16) = B_362;
        jcb(index,17) = B_386+ B_478;
        jcb(index,18) = B_388+ B_479;
        jcb(index,19) = 2*B_319+ 2*B_321+ 2*B_463;
        jcb(index,20) = 0.9*B_315+ B_317;
        jcb(index,21) = B_314+ 0.9*B_316+ 2*B_320+ B_328+ B_330;
        jcb(index,22) = B_318+ 2*B_322+ B_363+ B_369+ B_371+ B_387+ B_389+ B_391;
        jcb(index,23) = 2*B_476;
        jcb(index,24) = 3*B_474;
        jcb(index,25) = 0;
        jcb(index,26) = 2*B_327+ 2*B_465;
        jcb(index,27) = B_329+ B_464;
        jcb(index,28) = 2*B_328+ B_330;
        jcb(index,29) = 0;
        jcb(index,30) = B_465;
        jcb(index,31) = 2*B_464;
        jcb(index,32) = B_390;
        jcb(index,33) = 2*B_386;
        jcb(index,34) = B_388;
        jcb(index,35) = 0.09*B_315;
        jcb(index,36) = 0.09*B_316;
        jcb(index,37) = 2*B_387+ B_389+ B_391;
        jcb(index,38) = 0;
        jcb(index,39) = B_405;
        jcb(index,40) = 0.4*B_400;
        jcb(index,41) = 0.4*B_401;
        jcb(index,42) = B_406;
        jcb(index,43) = 0;
        jcb(index,44) = B_392;
        jcb(index,45) = B_393;
        jcb(index,46) = 0;
        jcb(index,47) = 2*B_483;
        jcb(index,48) = 0;
        jcb(index,49) = 2*B_483;
        jcb(index,50) = B_521;
        jcb(index,51) = B_522;
        jcb(index,52) = 0;
        jcb(index,53) = B_507;
        jcb(index,54) = B_510;
        jcb(index,55) = B_513+ B_515;
        jcb(index,56) = B_520;
        jcb(index,57) = B_514;
        jcb(index,58) = - B_505- B_507;
        jcb(index,59) = - B_506;
        jcb(index,60) = - B_508- B_510;
        jcb(index,61) = - B_509;
        jcb(index,62) = - B_511- B_513- B_515;
        jcb(index,63) = - B_512- B_514;
        jcb(index,64) = - B_516- B_518- B_520;
        jcb(index,65) = - B_517;
        jcb(index,66) = - B_519;
        jcb(index,67) = B_504;
        jcb(index,68) = 0;
        jcb(index,69) = 0;
        jcb(index,70) = B_22;
        jcb(index,71) = B_23;
        jcb(index,72) = 0;
        jcb(index,73) = B_33;
        jcb(index,74) = B_34;
        jcb(index,75) = 0;
        jcb(index,76) = 2*B_454;
        jcb(index,77) = B_319;
        jcb(index,78) = B_41+ B_43;
        jcb(index,79) = B_315;
        jcb(index,80) = B_481+ 3*B_483+ 2*B_490;
        jcb(index,81) = B_93;
        jcb(index,82) = B_100;
        jcb(index,83) = B_79+ B_89+ B_91;
        jcb(index,84) = B_12;
        jcb(index,85) = B_108;
        jcb(index,86) = B_134;
        jcb(index,87) = B_498;
        jcb(index,88) = B_254+ B_258;
        jcb(index,89) = B_180+ B_184;
        jcb(index,90) = B_226;
        jcb(index,91) = B_457+ B_484+ B_500;
        jcb(index,92) = B_486+ B_496;
        jcb(index,93) = B_142;
        jcb(index,94) = B_343+ B_468+ B_492+ B_502;
        jcb(index,95) = B_150;
        jcb(index,96) = 2*B_4+ B_13+ B_33+ B_42+ B_44+ B_109+ B_316+ B_320;
        jcb(index,97) = B_162;
        jcb(index,98) = B_10;
        jcb(index,99) = B_493+ B_497+ B_501;
        jcb(index,100) = 2*B_5+ 2*B_6+ B_11+ B_16+ B_22+ B_80+ B_94+ B_135+ B_181+ B_227+ B_255;
        jcb(index,101) = B_118+ B_311+ B_360;
        jcb(index,102) = B_14+ B_17+ B_288;
        jcb(index,103) = B_34+ B_482;
        jcb(index,104) = B_126;
        jcb(index,105) = 2*B_7+ B_15+ B_20+ 2*B_49+ 2*B_273+ 2*B_333+ B_344;
        jcb(index,106) = 2*B_334+ 2*B_335+ 2*B_336+ B_361+ B_378+ 2*B_380+ 2*B_382;
        jcb(index,107) = 2*B_274+ 2*B_275+ 2*B_276+ B_277+ B_289+ B_312+ B_379+ 2*B_381+ 2*B_383;
        jcb(index,108) = 2*B_50+ B_90+ B_92+ B_101;
        jcb(index,109) = B_68+ B_119+ B_127+ B_143+ B_151+ B_163+ B_185+ B_259+ 2*B_422;
        jcb(index,110) = B_21+ B_23+ B_69;
        jcb(index,111) = B_485+ B_487+ 2*B_491+ B_499+ B_503;
        jcb(index,112) = 0;
        jcb(index,113) = 0.333333*B_498;
        jcb(index,114) = 0.5*B_500;
        jcb(index,115) = 0.333333*B_496;
        jcb(index,116) = B_343+ B_468+ B_492+ 0.5*B_502;
        jcb(index,117) = B_493+ 0.333333*B_497+ 0.5*B_501;
        jcb(index,118) = B_360;
        jcb(index,119) = 2*B_333+ B_344;
        jcb(index,120) = 2*B_334+ 2*B_335+ 2*B_336+ B_361+ 0.5*B_378+ B_380+ B_382;
        jcb(index,121) = 0.5*B_379+ B_381+ B_383;
        jcb(index,122) = 0.333333*B_499+ 0.5*B_503;
        jcb(index,123) = 0;
        jcb(index,124) = 2*B_454;
        jcb(index,125) = B_319;
        jcb(index,126) = B_315;
        jcb(index,127) = B_490;
        jcb(index,128) = 0.333333*B_498;
        jcb(index,129) = B_457+ B_484+ 0.5*B_500;
        jcb(index,130) = 0.5*B_486+ 0.333333*B_496;
        jcb(index,131) = 0.5*B_502;
        jcb(index,132) = B_316+ B_320;
        jcb(index,133) = 0.333333*B_497+ 0.5*B_501;
        jcb(index,134) = B_311;
        jcb(index,135) = B_288;
        jcb(index,136) = 2*B_273;
        jcb(index,137) = 0.5*B_378+ B_380+ B_382;
        jcb(index,138) = 2*B_274+ 2*B_275+ 2*B_276+ B_277+ B_289+ B_312+ 0.5*B_379+ B_381+ B_383;
        jcb(index,139) = B_485+ 0.5*B_487+ B_491+ 0.333333*B_499+ 0.5*B_503;
        jcb(index,140) = 0;
        jcb(index,141) = B_12;
        jcb(index,142) = B_13;
        jcb(index,143) = B_10;
        jcb(index,144) = B_11+ B_16+ B_22;
        jcb(index,145) = B_14+ B_17;
        jcb(index,146) = B_15+ B_20;
        jcb(index,147) = B_21+ B_23;
        jcb(index,148) = 0;
        jcb(index,149) = B_481+ 3*B_483+ B_490;
        jcb(index,150) = B_93;
        jcb(index,151) = B_100;
        jcb(index,152) = B_79+ B_89+ B_91;
        jcb(index,153) = 0.333333*B_498;
        jcb(index,154) = 0.5*B_486+ 0.333333*B_496;
        jcb(index,155) = 0.333333*B_497;
        jcb(index,156) = B_80+ B_94;
        jcb(index,157) = B_482;
        jcb(index,158) = 2*B_49;
        jcb(index,159) = 2*B_50+ B_90+ B_92+ B_101;
        jcb(index,160) = B_68+ 2*B_422;
        jcb(index,161) = B_69;
        jcb(index,162) = 0.5*B_487+ B_491+ 0.333333*B_499;
        jcb(index,163) = 0;
        jcb(index,164) = B_41+ B_43;
        jcb(index,165) = B_108;
        jcb(index,166) = 2*B_4+ B_33+ B_42+ B_44+ B_109;
        jcb(index,167) = 2*B_5+ 2*B_6;
        jcb(index,168) = B_34;
        jcb(index,169) = 2*B_7;
        jcb(index,170) = 0;
        jcb(index,171) = B_134;
        jcb(index,172) = B_254+ B_258;
        jcb(index,173) = B_180+ B_184;
        jcb(index,174) = B_226;
        jcb(index,175) = B_142;
        jcb(index,176) = B_150;
        jcb(index,177) = B_162;
        jcb(index,178) = B_135+ B_181+ B_227+ B_255;
        jcb(index,179) = B_118;
        jcb(index,180) = B_126;
        jcb(index,181) = B_119+ B_127+ B_143+ B_151+ B_163+ B_185+ B_259;
        jcb(index,182) = 0;
        jcb(index,183) = B_16;
        jcb(index,184) = B_17;
        jcb(index,185) = 0;
        jcb(index,186) = B_62;
        jcb(index,187) = B_63;
        jcb(index,188) = B_476;
        jcb(index,189) = B_474;
        jcb(index,190) = 0;
        jcb(index,191) = B_362+ B_471;
        jcb(index,192) = B_363;
        jcb(index,193) = B_476;
        jcb(index,194) = 0;
        jcb(index,195) = 4*B_313+ 4*B_462;
        jcb(index,196) = 2*B_327+ 2*B_465;
        jcb(index,197) = 3*B_329+ 3*B_464;
        jcb(index,198) = 3*B_319+ 3*B_321+ 3*B_463;
        jcb(index,199) = B_315+ B_317+ B_461;
        jcb(index,200) = 4*B_314+ B_316+ 3*B_320+ 2*B_328+ 3*B_330;
        jcb(index,201) = B_318+ 3*B_322;
        jcb(index,202) = 0;
        jcb(index,203) = B_116;
        jcb(index,204) = B_117;
        jcb(index,205) = B_469;
        jcb(index,206) = 0;
        jcb(index,207) = B_458;
        jcb(index,208) = B_455;
        jcb(index,209) = B_37+ B_47;
        jcb(index,210) = B_418;
        jcb(index,211) = 0.4*B_400;
        jcb(index,212) = 0.333*B_426;
        jcb(index,213) = B_70;
        jcb(index,214) = B_188;
        jcb(index,215) = B_204;
        jcb(index,216) = B_245;
        jcb(index,217) = B_345;
        jcb(index,218) = B_72;
        jcb(index,219) = B_222;
        jcb(index,220) = B_262;
        jcb(index,221) = B_232;
        jcb(index,222) = B_394+ B_396+ B_407+ B_409;
        jcb(index,223) = B_196;
        jcb(index,224) = B_140;
        jcb(index,225) = B_156+ B_158;
        jcb(index,226) = B_28;
        jcb(index,227) = B_116;
        jcb(index,228) = B_71+ B_73+ B_346+ B_395+ B_397+ 0.4*B_401;
        jcb(index,229) = B_284+ B_408;
        jcb(index,230) = B_410;
        jcb(index,231) = B_48+ B_62+ B_117+ B_141+ B_159+ B_189+ B_197+ B_205+ B_223+ B_233+ B_246+ B_263+ B_420;
        jcb(index,232) = B_29+ B_63+ B_157+ B_285;
        jcb(index,233) = 0;
        jcb(index,234) = B_188;
        jcb(index,235) = B_204;
        jcb(index,236) = B_245;
        jcb(index,237) = B_222;
        jcb(index,238) = B_262;
        jcb(index,239) = B_232;
        jcb(index,240) = B_196;
        jcb(index,241) = B_140;
        jcb(index,242) = B_158;
        jcb(index,243) = B_141+ B_159+ B_189+ B_197+ B_205+ B_223+ B_233+ B_246+ B_263;
        jcb(index,244) = 0;
        jcb(index,245) = 2*B_370+ 2*B_472;
        jcb(index,246) = 3*B_368+ 3*B_473;
        jcb(index,247) = B_390+ B_477;
        jcb(index,248) = B_386+ B_478;
        jcb(index,249) = 2*B_388+ 2*B_479;
        jcb(index,250) = 3*B_369+ 2*B_371+ B_387+ 2*B_389+ B_391;
        jcb(index,251) = 0;
        jcb(index,252) = B_477;
        jcb(index,253) = 2*B_478;
        jcb(index,254) = B_479;
        jcb(index,255) = - B_448;
        jcb(index,256) = 0.8*B_247;
        jcb(index,257) = 0.8*B_248;
        jcb(index,258) = - B_279- B_454;
        jcb(index,259) = B_278;
        jcb(index,260) = - B_216;
        jcb(index,261) = - B_217;
        jcb(index,262) = - B_313- B_462;
        jcb(index,263) = - B_314;
        jcb(index,264) = - B_327- B_465;
        jcb(index,265) = - B_328;
        jcb(index,266) = - B_329- B_464;
        jcb(index,267) = - B_330;
        jcb(index,268) = - B_370- B_472;
        jcb(index,269) = - B_371;
        jcb(index,270) = - B_368- B_473;
        jcb(index,271) = - B_369;
        jcb(index,272) = - B_405;
        jcb(index,273) = B_403;
        jcb(index,274) = B_404;
        jcb(index,275) = - B_406;
        jcb(index,276) = - B_77;
        jcb(index,277) = - B_78;
        jcb(index,278) = - B_132;
        jcb(index,279) = - B_133;
        jcb(index,280) = - B_178;
        jcb(index,281) = - B_179;
        jcb(index,282) = - B_458;
        jcb(index,283) = B_490;
        jcb(index,284) = B_491;
        jcb(index,285) = - B_455;
        jcb(index,286) = B_378;
        jcb(index,287) = B_277+ B_379;
        jcb(index,288) = - B_390- B_477;
        jcb(index,289) = - B_391;
        jcb(index,290) = - B_362- B_471;
        jcb(index,291) = - B_363;
        jcb(index,292) = - B_386- B_478;
        jcb(index,293) = - B_387;
        jcb(index,294) = - B_388- B_479;
        jcb(index,295) = - B_389;
        jcb(index,296) = - B_392;
        jcb(index,297) = 0.6*B_400;
        jcb(index,298) = B_402;
        jcb(index,299) = - B_393+ 0.6*B_401;
        jcb(index,300) = - B_319- B_321- B_463;
        jcb(index,301) = - B_320;
        jcb(index,302) = - B_322;
        jcb(index,303) = - B_173- B_435;
        jcb(index,304) = B_269;
        jcb(index,305) = - B_174+ B_270;
        jcb(index,306) = - B_37- B_47- B_53;
        jcb(index,307) = - B_48+ B_420;
        jcb(index,308) = - B_54;
        jcb(index,309) = B_53;
        jcb(index,310) = - B_41- B_43- B_418;
        jcb(index,311) = B_89;
        jcb(index,312) = - B_42- B_44;
        jcb(index,313) = 0;
        jcb(index,314) = B_54+ B_90;
        jcb(index,315) = - B_104;
        jcb(index,316) = B_98;
        jcb(index,317) = B_99;
        jcb(index,318) = - B_105;
        jcb(index,319) = - B_214- B_442;
        jcb(index,320) = 0.04*B_188;
        jcb(index,321) = - B_215;
        jcb(index,322) = 0.04*B_189;
        jcb(index,323) = - B_171- B_434;
        jcb(index,324) = B_154;
        jcb(index,325) = - B_172;
        jcb(index,326) = B_155;
        jcb(index,327) = - B_251- B_253- B_450;
        jcb(index,328) = B_234;
        jcb(index,329) = - B_252;
        jcb(index,330) = B_235;
        jcb(index,331) = - B_400;
        jcb(index,332) = B_396+ B_411;
        jcb(index,333) = B_397- B_401;
        jcb(index,334) = B_412;
        jcb(index,335) = - B_267- B_451;
        jcb(index,336) = B_260;
        jcb(index,337) = - B_268;
        jcb(index,338) = B_261;
        jcb(index,339) = - B_198;
        jcb(index,340) = B_194;
        jcb(index,341) = - B_199;
        jcb(index,342) = B_195;
        jcb(index,343) = - B_247- B_447;
        jcb(index,344) = B_243;
        jcb(index,345) = - B_248;
        jcb(index,346) = B_244;
        jcb(index,347) = - B_192- B_437;
        jcb(index,348) = B_186;
        jcb(index,349) = - B_193;
        jcb(index,350) = B_187;
        jcb(index,351) = B_104;
        jcb(index,352) = - B_98- B_102;
        jcb(index,353) = B_95;
        jcb(index,354) = - B_99;
        jcb(index,355) = - B_103+ B_105;
        jcb(index,356) = - B_146- B_432;
        jcb(index,357) = B_138;
        jcb(index,358) = - B_147;
        jcb(index,359) = B_139;
        jcb(index,360) = - B_208- B_441;
        jcb(index,361) = B_202;
        jcb(index,362) = - B_209;
        jcb(index,363) = B_203;
        jcb(index,364) = - B_74- B_75- B_426;
        jcb(index,365) = - B_76;
        jcb(index,366) = B_66;
        jcb(index,367) = B_67;
        jcb(index,368) = - B_152;
        jcb(index,369) = 0.18*B_168;
        jcb(index,370) = B_156+ B_166+ 0.18*B_169;
        jcb(index,371) = B_167;
        jcb(index,372) = - B_153;
        jcb(index,373) = B_157;
        jcb(index,374) = - 0.9*B_315- B_317- B_461;
        jcb(index,375) = - 0.9*B_316;
        jcb(index,376) = - B_318;
        jcb(index,377) = - B_70- B_424;
        jcb(index,378) = B_100;
        jcb(index,379) = B_60- B_71;
        jcb(index,380) = B_61;
        jcb(index,381) = B_101;
        jcb(index,382) = - B_175- B_177- B_436;
        jcb(index,383) = B_160;
        jcb(index,384) = - B_176;
        jcb(index,385) = B_161;
        jcb(index,386) = - B_130;
        jcb(index,387) = 0.23125*B_134;
        jcb(index,388) = 0.28*B_254;
        jcb(index,389) = 0.22*B_180;
        jcb(index,390) = 0.45*B_226;
        jcb(index,391) = 0.23125*B_135+ 0.22*B_181+ 0.45*B_227+ 0.28*B_255;
        jcb(index,392) = - B_131;
        jcb(index,393) = - B_224- B_443;
        jcb(index,394) = B_220;
        jcb(index,395) = - B_225;
        jcb(index,396) = B_221;
        jcb(index,397) = - B_374- B_453;
        jcb(index,398) = B_384;
        jcb(index,399) = B_484;
        jcb(index,400) = B_303+ B_486;
        jcb(index,401) = B_304+ B_385;
        jcb(index,402) = - B_375;
        jcb(index,403) = B_275;
        jcb(index,404) = B_485+ B_487;
        jcb(index,405) = - B_402- B_403;
        jcb(index,406) = B_394+ B_398+ B_407+ B_409;
        jcb(index,407) = - B_404;
        jcb(index,408) = B_395;
        jcb(index,409) = B_408;
        jcb(index,410) = B_410;
        jcb(index,411) = B_399;
        jcb(index,412) = - B_239- B_445;
        jcb(index,413) = B_230;
        jcb(index,414) = - B_240;
        jcb(index,415) = B_231;
        jcb(index,416) = - B_59- B_423- B_481- B_483- B_490;
        jcb(index,417) = - B_482;
        jcb(index,418) = B_57;
        jcb(index,419) = B_58;
        jcb(index,420) = - B_491;
        jcb(index,421) = - B_93- B_95;
        jcb(index,422) = B_79+ B_81+ B_91;
        jcb(index,423) = B_80- B_94;
        jcb(index,424) = B_92;
        jcb(index,425) = B_82;
        jcb(index,426) = 0.85*B_224+ 0.67*B_443;
        jcb(index,427) = - B_241- B_446;
        jcb(index,428) = 0.88*B_218+ 0.56*B_222;
        jcb(index,429) = B_249+ 0.67*B_449;
        jcb(index,430) = 0.88*B_219;
        jcb(index,431) = 0.85*B_225- B_242+ B_250;
        jcb(index,432) = 0.56*B_223;
        jcb(index,433) = 0;
        jcb(index,434) = B_214+ B_442;
        jcb(index,435) = 0.7*B_192+ B_437;
        jcb(index,436) = - B_200- B_438;
        jcb(index,437) = 0.96*B_188+ B_190;
        jcb(index,438) = B_191;
        jcb(index,439) = 0.7*B_193- B_201+ B_215;
        jcb(index,440) = 0.96*B_189;
        jcb(index,441) = 0;
        jcb(index,442) = - B_98+ B_102;
        jcb(index,443) = 0;
        jcb(index,444) = - B_96- B_99- B_100- B_106;
        jcb(index,445) = B_83;
        jcb(index,446) = 0;
        jcb(index,447) = - B_97+ B_103;
        jcb(index,448) = - B_101;
        jcb(index,449) = B_84;
        jcb(index,450) = - B_35- B_286- B_417;
        jcb(index,451) = 0.13875*B_134;
        jcb(index,452) = 0.09*B_254;
        jcb(index,453) = 0.13875*B_135+ 0.09*B_255;
        jcb(index,454) = - B_36;
        jcb(index,455) = - B_287;
        jcb(index,456) = B_32;
        jcb(index,457) = - B_112;
        jcb(index,458) = 0.2*B_190;
        jcb(index,459) = 0.5*B_206;
        jcb(index,460) = 0.18*B_218;
        jcb(index,461) = 0.03*B_180;
        jcb(index,462) = 0.25*B_264;
        jcb(index,463) = 0.25*B_236;
        jcb(index,464) = 0.25*B_144;
        jcb(index,465) = 0.03*B_181;
        jcb(index,466) = B_121+ 0.25*B_145+ 0.2*B_191+ 0.5*B_207+ 0.18*B_219+ 0.25*B_237+ 0.25*B_265;
        jcb(index,467) = - B_113;
        jcb(index,468) = B_374;
        jcb(index,469) = - B_372- B_384- B_475;
        jcb(index,470) = B_376;
        jcb(index,471) = B_498;
        jcb(index,472) = B_500;
        jcb(index,473) = B_496;
        jcb(index,474) = B_502;
        jcb(index,475) = B_497+ B_501;
        jcb(index,476) = B_377- B_385;
        jcb(index,477) = - B_373+ B_375;
        jcb(index,478) = B_382;
        jcb(index,479) = B_383;
        jcb(index,480) = B_499+ B_503;
        jcb(index,481) = - B_269- B_452;
        jcb(index,482) = B_258;
        jcb(index,483) = 0.044*B_262;
        jcb(index,484) = - B_270;
        jcb(index,485) = 0.044*B_263;
        jcb(index,486) = B_259;
        jcb(index,487) = B_77;
        jcb(index,488) = B_93;
        jcb(index,489) = - B_79- B_81- B_83- B_85- B_87- B_89- B_91;
        jcb(index,490) = - B_80+ B_94;
        jcb(index,491) = B_78;
        jcb(index,492) = - B_86- B_88;
        jcb(index,493) = - B_90- B_92;
        jcb(index,494) = - B_82- B_84;
        jcb(index,495) = 0.82*B_178;
        jcb(index,496) = 0.3*B_192;
        jcb(index,497) = - B_186- B_188- B_190;
        jcb(index,498) = - B_191;
        jcb(index,499) = 0.82*B_179+ 0.3*B_193;
        jcb(index,500) = - B_189;
        jcb(index,501) = - B_187;
        jcb(index,502) = 0.3*B_208;
        jcb(index,503) = B_200;
        jcb(index,504) = 0;
        jcb(index,505) = - B_202- B_204- B_206;
        jcb(index,506) = - B_207;
        jcb(index,507) = B_201+ 0.3*B_209;
        jcb(index,508) = - B_205;
        jcb(index,509) = - B_203;
        jcb(index,510) = B_173+ B_435;
        jcb(index,511) = B_175;
        jcb(index,512) = 0.25*B_445;
        jcb(index,513) = 0;
        jcb(index,514) = - B_128;
        jcb(index,515) = B_212+ B_440;
        jcb(index,516) = B_431;
        jcb(index,517) = 0.63*B_134;
        jcb(index,518) = 0.14*B_254;
        jcb(index,519) = 0.31*B_180;
        jcb(index,520) = 0;
        jcb(index,521) = 0.22*B_226+ B_444;
        jcb(index,522) = 0.25*B_232+ 0.125*B_236+ 0.5*B_238;
        jcb(index,523) = B_433;
        jcb(index,524) = 0;
        jcb(index,525) = 0.63*B_135+ 0.31*B_181+ 0.22*B_227+ 0.14*B_255;
        jcb(index,526) = 0.125*B_237;
        jcb(index,527) = B_124- B_129+ B_174+ B_176+ B_213;
        jcb(index,528) = B_307;
        jcb(index,529) = B_354;
        jcb(index,530) = B_125+ B_126+ B_308+ B_355+ B_428+ B_429;
        jcb(index,531) = 0.25*B_233;
        jcb(index,532) = 0;
        jcb(index,533) = B_127;
        jcb(index,534) = 0;
        jcb(index,535) = 0.7*B_208;
        jcb(index,536) = 0.5*B_445;
        jcb(index,537) = 0.5*B_206;
        jcb(index,538) = - B_212- B_440;
        jcb(index,539) = 0.04*B_180;
        jcb(index,540) = B_210;
        jcb(index,541) = 0.25*B_264;
        jcb(index,542) = 0.9*B_226;
        jcb(index,543) = 0.5*B_232+ 0.5*B_236+ B_238;
        jcb(index,544) = 0.04*B_181+ 0.9*B_227;
        jcb(index,545) = 0.5*B_207+ 0.5*B_237+ 0.25*B_265;
        jcb(index,546) = 0.7*B_209+ B_211- B_213;
        jcb(index,547) = 0.5*B_233;
        jcb(index,548) = 0;
        jcb(index,549) = - B_12- B_18- B_280;
        jcb(index,550) = 0.05*B_108+ 0.69*B_431;
        jcb(index,551) = - B_13+ 0.05*B_109;
        jcb(index,552) = B_26;
        jcb(index,553) = - B_19;
        jcb(index,554) = - B_281;
        jcb(index,555) = B_428;
        jcb(index,556) = B_27;
        jcb(index,557) = - B_108- B_110- B_305- B_431;
        jcb(index,558) = 0.06*B_180;
        jcb(index,559) = - B_109;
        jcb(index,560) = 0.06*B_181;
        jcb(index,561) = - B_111;
        jcb(index,562) = - B_306;
        jcb(index,563) = 0.2*B_247;
        jcb(index,564) = B_241;
        jcb(index,565) = - B_243- B_245;
        jcb(index,566) = 0;
        jcb(index,567) = 0;
        jcb(index,568) = 0;
        jcb(index,569) = B_242+ 0.2*B_248;
        jcb(index,570) = - B_246;
        jcb(index,571) = - B_244;
        jcb(index,572) = B_372;
        jcb(index,573) = - B_345- B_376- B_466;
        jcb(index,574) = B_347;
        jcb(index,575) = 0;
        jcb(index,576) = 0;
        jcb(index,577) = B_492;
        jcb(index,578) = B_493;
        jcb(index,579) = - B_346;
        jcb(index,580) = - B_377;
        jcb(index,581) = B_348+ B_373;
        jcb(index,582) = B_336;
        jcb(index,583) = 0;
        jcb(index,584) = 0;
        jcb(index,585) = 2*B_481+ B_490;
        jcb(index,586) = - B_72- B_425;
        jcb(index,587) = B_494+ B_498;
        jcb(index,588) = B_398;
        jcb(index,589) = B_486+ B_488+ B_496;
        jcb(index,590) = B_150;
        jcb(index,591) = B_497;
        jcb(index,592) = B_64- B_73;
        jcb(index,593) = 2*B_482+ B_489+ B_495;
        jcb(index,594) = B_126;
        jcb(index,595) = B_65;
        jcb(index,596) = B_127+ B_151+ B_399;
        jcb(index,597) = B_487+ B_491+ B_499;
        jcb(index,598) = B_216;
        jcb(index,599) = 0.15*B_224;
        jcb(index,600) = - B_218- B_220- B_222;
        jcb(index,601) = - B_219;
        jcb(index,602) = B_217+ 0.15*B_225;
        jcb(index,603) = - B_223;
        jcb(index,604) = - B_221;
        jcb(index,605) = - B_134- B_136- B_323- B_364;
        jcb(index,606) = - B_135;
        jcb(index,607) = - B_137;
        jcb(index,608) = - B_324;
        jcb(index,609) = - B_365;
        jcb(index,610) = - B_122- B_309- B_356- B_427;
        jcb(index,611) = B_114;
        jcb(index,612) = - B_123;
        jcb(index,613) = - B_310;
        jcb(index,614) = - B_357;
        jcb(index,615) = B_115;
        jcb(index,616) = - B_347- B_353- B_470- B_494- B_498;
        jcb(index,617) = - B_495;
        jcb(index,618) = - B_348;
        jcb(index,619) = B_351;
        jcb(index,620) = B_352;
        jcb(index,621) = - B_499;
        jcb(index,622) = - B_254- B_256- B_258;
        jcb(index,623) = - B_255;
        jcb(index,624) = - B_257;
        jcb(index,625) = - B_259;
        jcb(index,626) = - B_180- B_182- B_184;
        jcb(index,627) = - B_181;
        jcb(index,628) = - B_183;
        jcb(index,629) = - B_185;
        jcb(index,630) = B_251+ B_450;
        jcb(index,631) = 0.5*B_198;
        jcb(index,632) = 0.25*B_445;
        jcb(index,633) = B_269;
        jcb(index,634) = 0.2*B_206;
        jcb(index,635) = 0;
        jcb(index,636) = - B_210- B_439;
        jcb(index,637) = 0.25*B_264;
        jcb(index,638) = 0.25*B_232+ 0.375*B_236+ B_238;
        jcb(index,639) = 0;
        jcb(index,640) = 0;
        jcb(index,641) = 0.2*B_207+ 0.375*B_237+ 0.25*B_265;
        jcb(index,642) = 0.5*B_199- B_211+ B_252+ B_270;
        jcb(index,643) = 0.25*B_233;
        jcb(index,644) = 0;
        jcb(index,645) = 0;
        jcb(index,646) = 0;
        jcb(index,647) = B_256;
        jcb(index,648) = - B_260- B_262- B_264- 2*B_266;
        jcb(index,649) = 0;
        jcb(index,650) = - B_265;
        jcb(index,651) = B_257;
        jcb(index,652) = - B_263;
        jcb(index,653) = 0;
        jcb(index,654) = - B_261;
        jcb(index,655) = B_267+ B_451;
        jcb(index,656) = B_452;
        jcb(index,657) = 0.65*B_254;
        jcb(index,658) = 0.956*B_262+ 0.5*B_264+ 2*B_266;
        jcb(index,659) = - B_226- B_228- B_444;
        jcb(index,660) = - B_227+ 0.65*B_255;
        jcb(index,661) = 0.5*B_265;
        jcb(index,662) = - B_229+ B_268;
        jcb(index,663) = 0.956*B_263;
        jcb(index,664) = 0;
        jcb(index,665) = 0;
        jcb(index,666) = 0.015*B_245;
        jcb(index,667) = 0.16*B_222;
        jcb(index,668) = B_184;
        jcb(index,669) = - B_249- B_449;
        jcb(index,670) = 0.02*B_196;
        jcb(index,671) = 0;
        jcb(index,672) = 0;
        jcb(index,673) = - B_250;
        jcb(index,674) = 0.02*B_197+ 0.16*B_223+ 0.015*B_246;
        jcb(index,675) = B_185;
        jcb(index,676) = 0;
        jcb(index,677) = - B_294- B_457- B_484- B_500;
        jcb(index,678) = B_488;
        jcb(index,679) = - B_501;
        jcb(index,680) = - B_295;
        jcb(index,681) = B_489;
        jcb(index,682) = B_290;
        jcb(index,683) = B_291;
        jcb(index,684) = - B_485;
        jcb(index,685) = B_253;
        jcb(index,686) = B_239;
        jcb(index,687) = 0.1*B_254;
        jcb(index,688) = B_228;
        jcb(index,689) = - B_230- B_232- B_234- B_236- 2*B_238;
        jcb(index,690) = 0.1*B_255;
        jcb(index,691) = - B_237;
        jcb(index,692) = B_229+ B_240;
        jcb(index,693) = - B_233;
        jcb(index,694) = - B_235;
        jcb(index,695) = 0;
        jcb(index,696) = - B_231;
        jcb(index,697) = - B_394- B_396- B_398- B_407- B_409- B_411;
        jcb(index,698) = - B_395- B_397;
        jcb(index,699) = - B_408;
        jcb(index,700) = - B_410;
        jcb(index,701) = - B_412;
        jcb(index,702) = - B_399;
        jcb(index,703) = 0.5*B_198;
        jcb(index,704) = 0.666667*B_136+ 0.666667*B_323+ 0.666667*B_364;
        jcb(index,705) = B_182;
        jcb(index,706) = - B_194- B_196;
        jcb(index,707) = 0;
        jcb(index,708) = 0.666667*B_137+ B_183+ 0.5*B_199;
        jcb(index,709) = 0.666667*B_324;
        jcb(index,710) = 0.666667*B_365;
        jcb(index,711) = - B_197;
        jcb(index,712) = 0;
        jcb(index,713) = - B_195;
        jcb(index,714) = - B_300- B_301- B_303- B_459- B_460- B_486- B_488- B_496;
        jcb(index,715) = - B_497;
        jcb(index,716) = - B_304;
        jcb(index,717) = - B_489;
        jcb(index,718) = - B_302;
        jcb(index,719) = B_298;
        jcb(index,720) = B_299;
        jcb(index,721) = - B_487;
        jcb(index,722) = B_132;
        jcb(index,723) = 0.18*B_178;
        jcb(index,724) = 0.3*B_146;
        jcb(index,725) = 0.33*B_443;
        jcb(index,726) = B_446;
        jcb(index,727) = 0.12*B_218+ 0.28*B_222;
        jcb(index,728) = 0.06*B_180;
        jcb(index,729) = 0.33*B_449;
        jcb(index,730) = 0;
        jcb(index,731) = - B_138- B_140- B_142- B_144- B_168;
        jcb(index,732) = - B_169;
        jcb(index,733) = 0.06*B_181;
        jcb(index,734) = - B_145+ 0.12*B_219;
        jcb(index,735) = B_133+ 0.3*B_147+ 0.18*B_179;
        jcb(index,736) = 0;
        jcb(index,737) = 0;
        jcb(index,738) = - B_141+ 0.28*B_223;
        jcb(index,739) = - B_143;
        jcb(index,740) = - B_139;
        jcb(index,741) = B_345;
        jcb(index,742) = B_494;
        jcb(index,743) = 0;
        jcb(index,744) = 0;
        jcb(index,745) = - B_343- B_468- B_492- B_502;
        jcb(index,746) = - B_493;
        jcb(index,747) = B_358;
        jcb(index,748) = B_346;
        jcb(index,749) = 0;
        jcb(index,750) = B_495;
        jcb(index,751) = 0;
        jcb(index,752) = - B_344;
        jcb(index,753) = B_339+ B_359;
        jcb(index,754) = 0;
        jcb(index,755) = 0;
        jcb(index,756) = B_340;
        jcb(index,757) = - B_503;
        jcb(index,758) = B_447;
        jcb(index,759) = 0.7*B_146+ B_432;
        jcb(index,760) = 0.33*B_443;
        jcb(index,761) = 0.985*B_245;
        jcb(index,762) = 0.12*B_218+ 0.28*B_222;
        jcb(index,763) = 0.47*B_180;
        jcb(index,764) = 0.33*B_449;
        jcb(index,765) = 0.98*B_196;
        jcb(index,766) = B_140+ B_142+ 0.75*B_144+ B_168;
        jcb(index,767) = - B_148- B_150- B_325- B_366- B_433;
        jcb(index,768) = B_169;
        jcb(index,769) = 0.47*B_181;
        jcb(index,770) = 0.75*B_145+ 0.12*B_219;
        jcb(index,771) = 0.7*B_147- B_149;
        jcb(index,772) = - B_326;
        jcb(index,773) = - B_367;
        jcb(index,774) = B_141+ 0.98*B_197+ 0.28*B_223+ 0.985*B_246;
        jcb(index,775) = B_143- B_151;
        jcb(index,776) = 0;
        jcb(index,777) = - B_313;
        jcb(index,778) = - B_327;
        jcb(index,779) = - B_329;
        jcb(index,780) = - B_319;
        jcb(index,781) = - B_41- B_43+ B_418;
        jcb(index,782) = - B_315;
        jcb(index,783) = 0;
        jcb(index,784) = - B_12;
        jcb(index,785) = - B_108;
        jcb(index,786) = 0;
        jcb(index,787) = - B_0- B_4- B_13- B_33- B_39- B_42- B_44- B_109- B_314- B_316- B_320- B_328- B_330;
        jcb(index,788) = 0;
        jcb(index,789) = - B_5+ B_414;
        jcb(index,790) = 0;
        jcb(index,791) = 0;
        jcb(index,792) = - B_34;
        jcb(index,793) = 0;
        jcb(index,794) = 0;
        jcb(index,795) = 0;
        jcb(index,796) = 0;
        jcb(index,797) = 0;
        jcb(index,798) = 2*B_448;
        jcb(index,799) = B_171;
        jcb(index,800) = B_447;
        jcb(index,801) = B_441;
        jcb(index,802) = B_177+ B_436;
        jcb(index,803) = 0.25*B_445;
        jcb(index,804) = B_446;
        jcb(index,805) = B_438;
        jcb(index,806) = 0;
        jcb(index,807) = B_204+ 0.3*B_206;
        jcb(index,808) = B_212+ B_440;
        jcb(index,809) = 0.985*B_245;
        jcb(index,810) = 0;
        jcb(index,811) = 0.1*B_254;
        jcb(index,812) = 0.23*B_180;
        jcb(index,813) = B_439;
        jcb(index,814) = 0;
        jcb(index,815) = 0.1*B_226+ B_444;
        jcb(index,816) = 0;
        jcb(index,817) = 0.25*B_232+ 0.125*B_236;
        jcb(index,818) = 0;
        jcb(index,819) = - B_168;
        jcb(index,820) = B_148+ B_150+ B_325+ B_366;
        jcb(index,821) = - B_154- B_156- B_158- B_160- B_162- B_164- B_166- B_169- 2*B_170;
        jcb(index,822) = 0.23*B_181+ 0.1*B_227+ 0.1*B_255;
        jcb(index,823) = - B_165- B_167+ 0.3*B_207+ 0.125*B_237;
        jcb(index,824) = B_149+ B_172+ B_213;
        jcb(index,825) = B_326;
        jcb(index,826) = B_367;
        jcb(index,827) = - B_159+ B_205+ 0.25*B_233+ 0.985*B_246;
        jcb(index,828) = - B_161;
        jcb(index,829) = B_151- B_163;
        jcb(index,830) = - B_155- B_157;
        jcb(index,831) = 0.09*B_315;
        jcb(index,832) = B_128;
        jcb(index,833) = 0;
        jcb(index,834) = B_12+ B_18+ B_280;
        jcb(index,835) = 0.4*B_108+ 0.31*B_431;
        jcb(index,836) = 0;
        jcb(index,837) = 0;
        jcb(index,838) = 0;
        jcb(index,839) = 0;
        jcb(index,840) = 0;
        jcb(index,841) = 0;
        jcb(index,842) = 0;
        jcb(index,843) = 0;
        jcb(index,844) = 0;
        jcb(index,845) = B_13+ 0.4*B_109+ 0.09*B_316;
        jcb(index,846) = 0;
        jcb(index,847) = - B_8- B_10- B_24- B_26- B_28;
        jcb(index,848) = - B_11;
        jcb(index,849) = 0;
        jcb(index,850) = B_14+ B_19+ B_129;
        jcb(index,851) = B_281;
        jcb(index,852) = B_416;
        jcb(index,853) = 0;
        jcb(index,854) = B_429;
        jcb(index,855) = B_15;
        jcb(index,856) = 0;
        jcb(index,857) = 0;
        jcb(index,858) = 0;
        jcb(index,859) = - B_25- B_27- B_29;
        jcb(index,860) = B_456;
        jcb(index,861) = B_364;
        jcb(index,862) = B_356;
        jcb(index,863) = - B_500;
        jcb(index,864) = B_409;
        jcb(index,865) = - B_496;
        jcb(index,866) = - B_492;
        jcb(index,867) = B_366;
        jcb(index,868) = 0;
        jcb(index,869) = - B_341- B_493- B_497- B_501;
        jcb(index,870) = 0;
        jcb(index,871) = 0;
        jcb(index,872) = - B_342;
        jcb(index,873) = 0;
        jcb(index,874) = 0;
        jcb(index,875) = B_337+ B_354+ B_357+ B_365+ B_367+ B_410;
        jcb(index,876) = B_355;
        jcb(index,877) = 0;
        jcb(index,878) = 0;
        jcb(index,879) = 0;
        jcb(index,880) = 0;
        jcb(index,881) = 0;
        jcb(index,882) = 0;
        jcb(index,883) = B_338;
        jcb(index,884) = 0;
        jcb(index,885) = - B_403;
        jcb(index,886) = - B_93;
        jcb(index,887) = - B_79;
        jcb(index,888) = - B_134;
        jcb(index,889) = - B_254;
        jcb(index,890) = - B_180;
        jcb(index,891) = - B_226;
        jcb(index,892) = 0;
        jcb(index,893) = - B_4;
        jcb(index,894) = B_156;
        jcb(index,895) = - B_10;
        jcb(index,896) = - B_5- B_6- B_11- B_16- B_22- B_45- B_51- B_80- B_94- B_135- B_181- B_227- B_255- B_271- B_331- B_404 - B_414- B_415;
        jcb(index,897) = 0;
        jcb(index,898) = - B_17;
        jcb(index,899) = - B_272;
        jcb(index,900) = 0;
        jcb(index,901) = - B_332;
        jcb(index,902) = 0;
        jcb(index,903) = B_2- B_7;
        jcb(index,904) = 0;
        jcb(index,905) = - B_46;
        jcb(index,906) = - B_52;
        jcb(index,907) = 0;
        jcb(index,908) = - B_23+ B_157;
        jcb(index,909) = 0;
        jcb(index,910) = B_480;
        jcb(index,911) = B_471;
        jcb(index,912) = B_434;
        jcb(index,913) = 0.6*B_400;
        jcb(index,914) = B_152;
        jcb(index,915) = B_461;
        jcb(index,916) = B_402;
        jcb(index,917) = B_438;
        jcb(index,918) = - B_190;
        jcb(index,919) = - B_206;
        jcb(index,920) = 0.75*B_108+ B_110+ B_305;
        jcb(index,921) = - B_218;
        jcb(index,922) = 0.7*B_122+ B_356;
        jcb(index,923) = 0.08*B_254;
        jcb(index,924) = 0.07*B_180;
        jcb(index,925) = - B_264;
        jcb(index,926) = - B_236;
        jcb(index,927) = 0;
        jcb(index,928) = - B_144+ 0.82*B_168;
        jcb(index,929) = B_433;
        jcb(index,930) = 0.75*B_109;
        jcb(index,931) = B_158+ B_162- B_166+ 0.82*B_169+ 2*B_170;
        jcb(index,932) = 0;
        jcb(index,933) = 0.07*B_181+ 0.08*B_255;
        jcb(index,934) = - B_114- B_116- B_118- 2*B_120- 2*B_121- B_145- B_167- B_191- B_207- B_219- B_237- B_265- B_311- B_358 - B_360;
        jcb(index,935) = B_111+ 0.7*B_123+ B_153+ 0.6*B_401;
        jcb(index,936) = B_306;
        jcb(index,937) = 0;
        jcb(index,938) = B_357;
        jcb(index,939) = 0;
        jcb(index,940) = 0;
        jcb(index,941) = - B_359- B_361;
        jcb(index,942) = - B_117+ B_159;
        jcb(index,943) = - B_312;
        jcb(index,944) = 0;
        jcb(index,945) = - B_119+ B_163;
        jcb(index,946) = - B_115;
        jcb(index,947) = 0;
        jcb(index,948) = - B_216;
        jcb(index,949) = - B_370;
        jcb(index,950) = - B_368;
        jcb(index,951) = - B_77;
        jcb(index,952) = - B_132;
        jcb(index,953) = - B_178;
        jcb(index,954) = - B_390;
        jcb(index,955) = - B_362;
        jcb(index,956) = - B_386;
        jcb(index,957) = - B_388;
        jcb(index,958) = - B_392;
        jcb(index,959) = B_319- B_321;
        jcb(index,960) = - B_173;
        jcb(index,961) = - B_104;
        jcb(index,962) = - B_214;
        jcb(index,963) = - B_171+ B_434;
        jcb(index,964) = - B_251;
        jcb(index,965) = - B_400;
        jcb(index,966) = B_451;
        jcb(index,967) = - 0.5*B_198;
        jcb(index,968) = - 0.2*B_247+ B_447;
        jcb(index,969) = - 0.3*B_192+ B_437;
        jcb(index,970) = - B_102;
        jcb(index,971) = - 0.3*B_146+ B_432;
        jcb(index,972) = - 0.3*B_208+ B_441;
        jcb(index,973) = - B_75+ 0.333*B_426;
        jcb(index,974) = - B_152;
        jcb(index,975) = - B_317;
        jcb(index,976) = - B_70+ B_424;
        jcb(index,977) = - B_175;
        jcb(index,978) = - B_130;
        jcb(index,979) = - 0.15*B_224+ B_443;
        jcb(index,980) = 0;
        jcb(index,981) = - B_239+ B_445;
        jcb(index,982) = 0;
        jcb(index,983) = - B_241;
        jcb(index,984) = - B_200;
        jcb(index,985) = - B_96;
        jcb(index,986) = - B_35+ 2*B_417;
        jcb(index,987) = - B_112;
        jcb(index,988) = - B_269;
        jcb(index,989) = B_81+ B_85;
        jcb(index,990) = 0;
        jcb(index,991) = 0;
        jcb(index,992) = - B_128;
        jcb(index,993) = - B_212;
        jcb(index,994) = B_12- B_18;
        jcb(index,995) = 0.75*B_108- B_110;
        jcb(index,996) = 0;
        jcb(index,997) = - B_345;
        jcb(index,998) = - B_72+ B_425;
        jcb(index,999) = 0;
        jcb(index,1000) = 0.13*B_134- B_136;
        jcb(index,1001) = - 0.7*B_122+ B_309+ B_427;
        jcb(index,1002) = 0;
        jcb(index,1003) = 0.25*B_254- B_256;
        jcb(index,1004) = 0.33*B_180- B_182;
        jcb(index,1005) = - B_210;
        jcb(index,1006) = 0;
        jcb(index,1007) = 0.19*B_226- B_228;
        jcb(index,1008) = - B_249;
        jcb(index,1009) = - B_294+ B_457;
        jcb(index,1010) = 0;
        jcb(index,1011) = - B_394- B_396;
        jcb(index,1012) = 0;
        jcb(index,1013) = 0;
        jcb(index,1014) = 0;
        jcb(index,1015) = B_343+ B_468;
        jcb(index,1016) = - B_148;
        jcb(index,1017) = B_13+ 2*B_33+ 0.75*B_109+ B_320;
        jcb(index,1018) = 0;
        jcb(index,1019) = B_10+ 2*B_24;
        jcb(index,1020) = - B_341;
        jcb(index,1021) = B_11- B_16+ B_22+ 0.13*B_135+ 0.33*B_181+ 0.19*B_227+ 0.25*B_255;
        jcb(index,1022) = 0;
        jcb(index,1023) = - B_14- B_17- B_19- B_30- B_36- B_60- B_64- B_71- B_73- B_76- B_78- B_97- B_103- B_105- B_111- B_113- 0.7 *B_123- B_124- B_129- B_131- B_133- B_137- 0.3*B_147- B_149- B_153- B_172- B_174- B_176- B_179- B_183- 0.3 *B_193- 0.5*B_199- B_201- 0.3*B_209- B_211- B_213- B_215- B_217- 0.15*B_225- B_229- B_240- B_242- 0.2 *B_248- B_250- B_252- B_257- B_270- B_288- B_292- B_295- B_318- B_322- B_342- B_346- B_363- B_369- B_371 - B_387- B_389- B_391- B_393- B_395- B_397- B_401;
        jcb(index,1024) = B_284+ B_310;
        jcb(index,1025) = 2*B_34+ B_416;
        jcb(index,1026) = 0;
        jcb(index,1027) = - B_125;
        jcb(index,1028) = - B_15+ B_20+ B_344;
        jcb(index,1029) = 0;
        jcb(index,1030) = - B_61+ B_62+ B_86;
        jcb(index,1031) = - B_289;
        jcb(index,1032) = - B_65;
        jcb(index,1033) = B_68;
        jcb(index,1034) = B_21+ B_23+ 2*B_25- B_31+ B_63+ B_69+ B_82+ B_285;
        jcb(index,1035) = - B_293;
        jcb(index,1036) = B_476;
        jcb(index,1037) = 2*B_454;
        jcb(index,1038) = 3*B_313+ 4*B_462;
        jcb(index,1039) = B_327+ B_465;
        jcb(index,1040) = 2*B_329+ B_464;
        jcb(index,1041) = B_458;
        jcb(index,1042) = B_477;
        jcb(index,1043) = 2*B_478;
        jcb(index,1044) = B_479;
        jcb(index,1045) = 3*B_319+ 3*B_321+ 3*B_463;
        jcb(index,1046) = 0.35*B_315+ B_317+ B_461;
        jcb(index,1047) = B_374+ 2*B_453;
        jcb(index,1048) = 0;
        jcb(index,1049) = - B_286;
        jcb(index,1050) = B_372- B_384+ B_475;
        jcb(index,1051) = - B_280;
        jcb(index,1052) = - B_305;
        jcb(index,1053) = - B_376;
        jcb(index,1054) = - B_323;
        jcb(index,1055) = - B_309;
        jcb(index,1056) = 0;
        jcb(index,1057) = 0;
        jcb(index,1058) = 0;
        jcb(index,1059) = B_457;
        jcb(index,1060) = - B_407;
        jcb(index,1061) = - B_303+ B_459;
        jcb(index,1062) = 0;
        jcb(index,1063) = - B_325;
        jcb(index,1064) = 3*B_314+ 0.35*B_316+ 3*B_320+ B_328+ 2*B_330;
        jcb(index,1065) = 0;
        jcb(index,1066) = 0;
        jcb(index,1067) = 0;
        jcb(index,1068) = - B_271;
        jcb(index,1069) = B_311;
        jcb(index,1070) = 0.94*B_288+ B_292+ B_318+ 3*B_322;
        jcb(index,1071) = - B_272- B_281- B_282- B_284- B_287- B_304- B_306- B_307- B_310- B_324- B_326- B_377- B_385- B_408;
        jcb(index,1072) = 0;
        jcb(index,1073) = B_373+ B_375;
        jcb(index,1074) = - B_308;
        jcb(index,1075) = B_273;
        jcb(index,1076) = B_380;
        jcb(index,1077) = B_296;
        jcb(index,1078) = B_274+ 2*B_276+ B_277+ 0.94*B_289+ B_297+ B_312+ B_381;
        jcb(index,1079) = 0;
        jcb(index,1080) = 0;
        jcb(index,1081) = - B_283- B_285;
        jcb(index,1082) = B_293+ B_456;
        jcb(index,1083) = B_216;
        jcb(index,1084) = B_370;
        jcb(index,1085) = B_368;
        jcb(index,1086) = B_77;
        jcb(index,1087) = B_132;
        jcb(index,1088) = B_178;
        jcb(index,1089) = B_390;
        jcb(index,1090) = B_362;
        jcb(index,1091) = B_386;
        jcb(index,1092) = B_388;
        jcb(index,1093) = B_321;
        jcb(index,1094) = B_104;
        jcb(index,1095) = B_171;
        jcb(index,1096) = B_198;
        jcb(index,1097) = B_102;
        jcb(index,1098) = B_75;
        jcb(index,1099) = B_152;
        jcb(index,1100) = B_317;
        jcb(index,1101) = B_70;
        jcb(index,1102) = B_175;
        jcb(index,1103) = B_130;
        jcb(index,1104) = 0.85*B_224;
        jcb(index,1105) = - B_481;
        jcb(index,1106) = 0;
        jcb(index,1107) = B_200;
        jcb(index,1108) = B_96;
        jcb(index,1109) = B_35;
        jcb(index,1110) = B_83+ B_87+ B_89;
        jcb(index,1111) = 0;
        jcb(index,1112) = B_18;
        jcb(index,1113) = B_110+ 1.155*B_431;
        jcb(index,1114) = B_72;
        jcb(index,1115) = 0;
        jcb(index,1116) = 0;
        jcb(index,1117) = B_122;
        jcb(index,1118) = - B_494;
        jcb(index,1119) = 0;
        jcb(index,1120) = 0;
        jcb(index,1121) = 0;
        jcb(index,1122) = B_249;
        jcb(index,1123) = B_294+ B_484+ B_500;
        jcb(index,1124) = 0;
        jcb(index,1125) = 0;
        jcb(index,1126) = - B_488;
        jcb(index,1127) = 0;
        jcb(index,1128) = B_492+ B_502;
        jcb(index,1129) = B_148;
        jcb(index,1130) = - B_33;
        jcb(index,1131) = 0;
        jcb(index,1132) = B_28;
        jcb(index,1133) = B_341+ B_493+ B_501;
        jcb(index,1134) = 0;
        jcb(index,1135) = 0;
        jcb(index,1136) = B_19+ B_30+ B_36+ B_71+ B_73+ B_76+ B_78+ B_97+ B_103+ B_105+ B_111+ B_123+ B_124+ B_131+ B_133+ B_149 + B_153+ B_172+ B_176+ B_179+ B_199+ B_201+ B_217+ 0.85*B_225+ B_250+ B_292+ B_295+ B_318+ B_322+ B_342 + B_363+ B_369+ B_371+ B_387+ B_389+ B_391;
        jcb(index,1137) = 0;
        jcb(index,1138) = - B_34- B_416- B_482- B_489- B_495;
        jcb(index,1139) = 0;
        jcb(index,1140) = B_125;
        jcb(index,1141) = 0;
        jcb(index,1142) = 0;
        jcb(index,1143) = B_88;
        jcb(index,1144) = 0;
        jcb(index,1145) = B_90;
        jcb(index,1146) = 0;
        jcb(index,1147) = B_29+ B_31+ B_84;
        jcb(index,1148) = B_293+ B_485+ B_503;
        jcb(index,1149) = B_469;
        jcb(index,1150) = B_476;
        jcb(index,1151) = B_474;
        jcb(index,1152) = 2*B_370+ 2*B_472;
        jcb(index,1153) = 3*B_368+ 3*B_473;
        jcb(index,1154) = B_390+ B_477;
        jcb(index,1155) = B_362+ B_471;
        jcb(index,1156) = B_386+ B_478;
        jcb(index,1157) = 2*B_388+ 2*B_479;
        jcb(index,1158) = - B_374;
        jcb(index,1159) = - B_372+ B_384+ B_475;
        jcb(index,1160) = B_345+ B_376+ 2*B_466;
        jcb(index,1161) = - B_364;
        jcb(index,1162) = - B_356;
        jcb(index,1163) = - B_347+ 0.85*B_470;
        jcb(index,1164) = 0;
        jcb(index,1165) = - B_409+ B_411;
        jcb(index,1166) = 0;
        jcb(index,1167) = B_468;
        jcb(index,1168) = - B_366;
        jcb(index,1169) = 0;
        jcb(index,1170) = B_341;
        jcb(index,1171) = - B_331;
        jcb(index,1172) = B_360;
        jcb(index,1173) = B_342+ B_346+ B_363+ 3*B_369+ 2*B_371+ B_387+ 2*B_389+ B_391;
        jcb(index,1174) = B_377+ B_385;
        jcb(index,1175) = 0;
        jcb(index,1176) = - B_332- B_337- B_348- B_354- B_357- B_365- B_367- B_373- B_375- B_410;
        jcb(index,1177) = - B_355;
        jcb(index,1178) = B_333;
        jcb(index,1179) = B_334+ 2*B_335+ B_349+ B_361+ B_378+ B_380+ B_412+ B_467;
        jcb(index,1180) = B_350;
        jcb(index,1181) = B_379+ B_381;
        jcb(index,1182) = 0;
        jcb(index,1183) = 0;
        jcb(index,1184) = - B_338;
        jcb(index,1185) = 0;
        jcb(index,1186) = B_173+ B_435;
        jcb(index,1187) = B_400;
        jcb(index,1188) = B_451;
        jcb(index,1189) = B_441;
        jcb(index,1190) = B_175;
        jcb(index,1191) = 0.75*B_445;
        jcb(index,1192) = B_112;
        jcb(index,1193) = B_452;
        jcb(index,1194) = 0.8*B_190;
        jcb(index,1195) = B_204+ 0.8*B_206;
        jcb(index,1196) = 0.25*B_108;
        jcb(index,1197) = 0.68*B_218;
        jcb(index,1198) = 1.13875*B_134;
        jcb(index,1199) = 0.3*B_122+ B_309+ B_427;
        jcb(index,1200) = 0.58*B_254;
        jcb(index,1201) = 0.57*B_180;
        jcb(index,1202) = B_439;
        jcb(index,1203) = 0.956*B_262+ 1.25*B_264+ B_266;
        jcb(index,1204) = B_444;
        jcb(index,1205) = 0.75*B_232+ 1.125*B_236+ 0.5*B_238;
        jcb(index,1206) = B_394+ B_398+ B_407+ B_409;
        jcb(index,1207) = 0.98*B_196;
        jcb(index,1208) = 0.75*B_144;
        jcb(index,1209) = 0.25*B_109;
        jcb(index,1210) = B_164+ B_166;
        jcb(index,1211) = 0;
        jcb(index,1212) = 1.13875*B_135+ 0.57*B_181+ 0.58*B_255;
        jcb(index,1213) = B_116+ B_118+ 2*B_120+ B_121+ 0.75*B_145+ B_165+ B_167+ 0.8*B_191+ 0.8*B_207+ 0.68*B_219+ 1.125*B_237  + 1.25*B_265+ B_311+ B_358+ B_360;
        jcb(index,1214) = B_113+ 0.3*B_123- B_124+ B_174+ B_176+ B_395+ B_401;
        jcb(index,1215) = - B_307+ B_310+ B_408;
        jcb(index,1216) = 0;
        jcb(index,1217) = - B_354+ B_410;
        jcb(index,1218) = - B_125- B_126- B_308- B_355- B_428- B_429;
        jcb(index,1219) = 0;
        jcb(index,1220) = B_359+ B_361;
        jcb(index,1221) = B_117+ 0.98*B_197+ B_205+ 0.75*B_233+ 0.956*B_263;
        jcb(index,1222) = B_312;
        jcb(index,1223) = 0;
        jcb(index,1224) = B_119- B_127+ B_399;
        jcb(index,1225) = 0;
        jcb(index,1226) = 0;
        jcb(index,1227) = B_455;
        jcb(index,1228) = B_37+ B_47+ B_53;
        jcb(index,1229) = 0.1*B_315;
        jcb(index,1230) = - B_301;
        jcb(index,1231) = - B_343;
        jcb(index,1232) = B_0+ B_39+ 0.1*B_316;
        jcb(index,1233) = B_28;
        jcb(index,1234) = 0;
        jcb(index,1235) = - B_6+ B_415;
        jcb(index,1236) = 0;
        jcb(index,1237) = - B_14;
        jcb(index,1238) = 0;
        jcb(index,1239) = 0;
        jcb(index,1240) = 0;
        jcb(index,1241) = 0;
        jcb(index,1242) = - B_2- B_7- B_15- B_20- B_49- B_273- B_302- B_333- B_344;
        jcb(index,1243) = - B_334+ B_467;
        jcb(index,1244) = B_48+ B_420;
        jcb(index,1245) = - B_274;
        jcb(index,1246) = - B_50+ B_54+ B_419;
        jcb(index,1247) = B_421;
        jcb(index,1248) = - B_21+ B_29;
        jcb(index,1249) = 0;
        jcb(index,1250) = B_353+ 0.15*B_470;
        jcb(index,1251) = - B_411;
        jcb(index,1252) = B_343;
        jcb(index,1253) = 0;
        jcb(index,1254) = B_331;
        jcb(index,1255) = - B_358- B_360;
        jcb(index,1256) = 0;
        jcb(index,1257) = 0;
        jcb(index,1258) = 0;
        jcb(index,1259) = B_332;
        jcb(index,1260) = 0;
        jcb(index,1261) = - B_333+ B_344;
        jcb(index,1262) = - B_334- 2*B_335- 2*B_336- B_339- B_349- B_351- B_359- B_361- B_378- B_380- B_382- B_412- B_467;
        jcb(index,1263) = - B_350;
        jcb(index,1264) = - B_379- B_381- B_383;
        jcb(index,1265) = - B_352;
        jcb(index,1266) = 0;
        jcb(index,1267) = - B_340;
        jcb(index,1268) = 0;
        jcb(index,1269) = B_37- B_47;
        jcb(index,1270) = 2*B_41;
        jcb(index,1271) = B_98;
        jcb(index,1272) = B_424;
        jcb(index,1273) = 0;
        jcb(index,1274) = B_96+ B_99+ B_100+ B_106;
        jcb(index,1275) = - B_85- B_87+ B_91;
        jcb(index,1276) = - B_188;
        jcb(index,1277) = - B_204;
        jcb(index,1278) = - B_245;
        jcb(index,1279) = - B_222;
        jcb(index,1280) = - B_262;
        jcb(index,1281) = 0;
        jcb(index,1282) = - B_232;
        jcb(index,1283) = - B_196;
        jcb(index,1284) = - B_140;
        jcb(index,1285) = 2*B_42;
        jcb(index,1286) = - B_158;
        jcb(index,1287) = 0;
        jcb(index,1288) = - B_45;
        jcb(index,1289) = - B_116;
        jcb(index,1290) = - B_60+ B_97;
        jcb(index,1291) = 0;
        jcb(index,1292) = 0;
        jcb(index,1293) = 0;
        jcb(index,1294) = 0;
        jcb(index,1295) = B_49;
        jcb(index,1296) = - B_349;
        jcb(index,1297) = - B_46- B_48- B_55- B_61- B_62- B_86- B_88- B_117- B_141- B_159- B_189- B_197- B_205- B_223- B_233- B_246 - B_263- B_296- B_350- B_420;
        jcb(index,1298) = - B_297;
        jcb(index,1299) = B_50+ B_92+ B_101+ B_419;
        jcb(index,1300) = - B_56+ B_422;
        jcb(index,1301) = - B_63;
        jcb(index,1302) = 0;
        jcb(index,1303) = 2*B_279;
        jcb(index,1304) = B_313;
        jcb(index,1305) = B_327;
        jcb(index,1306) = B_329;
        jcb(index,1307) = B_455;
        jcb(index,1308) = 0.46*B_315;
        jcb(index,1309) = B_294;
        jcb(index,1310) = B_300+ B_301+ B_460;
        jcb(index,1311) = B_314+ 0.46*B_316+ B_328+ B_330;
        jcb(index,1312) = 0;
        jcb(index,1313) = 0;
        jcb(index,1314) = B_271;
        jcb(index,1315) = - B_311;
        jcb(index,1316) = - B_288+ B_295;
        jcb(index,1317) = B_272+ B_284;
        jcb(index,1318) = 0;
        jcb(index,1319) = 0;
        jcb(index,1320) = 0;
        jcb(index,1321) = - B_273+ B_302;
        jcb(index,1322) = - B_378- B_380- B_382;
        jcb(index,1323) = - B_296;
        jcb(index,1324) = - B_274- 2*B_275- 2*B_276- 2*B_277- 2*B_278- B_289- B_290- B_297- B_298- B_312- B_379- B_381- B_383;
        jcb(index,1325) = - B_299;
        jcb(index,1326) = 0;
        jcb(index,1327) = B_285- B_291;
        jcb(index,1328) = 0;
        jcb(index,1329) = B_469;
        jcb(index,1330) = B_458;
        jcb(index,1331) = B_173+ B_435;
        jcb(index,1332) = - B_53;
        jcb(index,1333) = B_214+ B_442;
        jcb(index,1334) = B_251+ B_253+ B_450;
        jcb(index,1335) = B_74+ B_75+ 0.667*B_426;
        jcb(index,1336) = B_70;
        jcb(index,1337) = B_175+ B_177+ B_436;
        jcb(index,1338) = B_59+ B_423;
        jcb(index,1339) = - B_100;
        jcb(index,1340) = B_452;
        jcb(index,1341) = - B_89- B_91;
        jcb(index,1342) = 0.96*B_188;
        jcb(index,1343) = B_204;
        jcb(index,1344) = 0.985*B_245;
        jcb(index,1345) = B_425;
        jcb(index,1346) = 0.84*B_222;
        jcb(index,1347) = B_353+ 0.15*B_470;
        jcb(index,1348) = 0;
        jcb(index,1349) = 0.956*B_262;
        jcb(index,1350) = B_249+ B_449;
        jcb(index,1351) = B_232- B_234;
        jcb(index,1352) = 0;
        jcb(index,1353) = 0.98*B_196;
        jcb(index,1354) = B_300+ B_460;
        jcb(index,1355) = B_140+ B_142;
        jcb(index,1356) = 0;
        jcb(index,1357) = B_158- B_160+ B_162;
        jcb(index,1358) = 0;
        jcb(index,1359) = B_45- B_51;
        jcb(index,1360) = B_116+ B_118;
        jcb(index,1361) = - B_64+ B_71+ B_76+ B_174+ B_176+ B_215+ B_250+ B_252;
        jcb(index,1362) = 0;
        jcb(index,1363) = 0;
        jcb(index,1364) = 0;
        jcb(index,1365) = 0;
        jcb(index,1366) = - B_49;
        jcb(index,1367) = B_349- B_351;
        jcb(index,1368) = B_46+ 2*B_55+ B_62+ B_117+ B_141+ B_159+ 0.96*B_189+ 0.98*B_197+ B_205+ 0.84*B_223+ B_233+ 0.985*B_246  + 0.956*B_263+ B_296+ B_350;
        jcb(index,1369) = B_297- B_298;
        jcb(index,1370) = - B_50- B_52- B_54- B_57- B_65- B_66- B_90- B_92- B_101- B_161- B_235- B_299- B_352- B_419;
        jcb(index,1371) = 2*B_56- B_58+ B_68+ B_119+ B_143+ B_163+ B_421;
        jcb(index,1372) = B_63- B_67+ B_69;
        jcb(index,1373) = 0;
        jcb(index,1374) = 0.333*B_426;
        jcb(index,1375) = B_59+ B_423;
        jcb(index,1376) = B_72;
        jcb(index,1377) = B_347+ 0.85*B_470;
        jcb(index,1378) = - B_258;
        jcb(index,1379) = - B_184;
        jcb(index,1380) = - B_398;
        jcb(index,1381) = B_301+ B_303+ B_459;
        jcb(index,1382) = - B_142;
        jcb(index,1383) = - B_150;
        jcb(index,1384) = - B_162;
        jcb(index,1385) = 0;
        jcb(index,1386) = B_51;
        jcb(index,1387) = - B_118;
        jcb(index,1388) = B_73;
        jcb(index,1389) = B_304;
        jcb(index,1390) = 0;
        jcb(index,1391) = B_348;
        jcb(index,1392) = - B_126;
        jcb(index,1393) = B_302;
        jcb(index,1394) = 0;
        jcb(index,1395) = - B_55;
        jcb(index,1396) = 0;
        jcb(index,1397) = B_52- B_57;
        jcb(index,1398) = - B_56- B_58- B_68- B_119- B_127- B_143- B_151- B_163- B_185- B_259- B_399- B_421- B_422;
        jcb(index,1399) = - B_69;
        jcb(index,1400) = 0;
        jcb(index,1401) = - B_405;
        jcb(index,1402) = B_392;
        jcb(index,1403) = B_442;
        jcb(index,1404) = 0.4*B_400;
        jcb(index,1405) = B_451;
        jcb(index,1406) = B_437;
        jcb(index,1407) = B_432;
        jcb(index,1408) = B_74+ 0.667*B_426;
        jcb(index,1409) = B_130;
        jcb(index,1410) = 0.67*B_443;
        jcb(index,1411) = 0;
        jcb(index,1412) = 0.75*B_445;
        jcb(index,1413) = B_106;
        jcb(index,1414) = B_35+ B_286;
        jcb(index,1415) = B_112;
        jcb(index,1416) = B_452;
        jcb(index,1417) = - B_81- B_83+ B_85;
        jcb(index,1418) = - B_186+ 0.96*B_188+ 0.8*B_190;
        jcb(index,1419) = - B_202+ 0.3*B_206;
        jcb(index,1420) = B_440;
        jcb(index,1421) = - B_243;
        jcb(index,1422) = 1.23*B_218- B_220+ 0.56*B_222;
        jcb(index,1423) = 0.13*B_134;
        jcb(index,1424) = B_427;
        jcb(index,1425) = 0.25*B_254;
        jcb(index,1426) = 0.26*B_180;
        jcb(index,1427) = B_210+ B_439;
        jcb(index,1428) = - B_260+ 0.956*B_262+ B_264+ B_266;
        jcb(index,1429) = 0.32*B_226+ B_444;
        jcb(index,1430) = 0.67*B_449;
        jcb(index,1431) = - B_230+ 0.75*B_232+ 0.875*B_236+ B_238;
        jcb(index,1432) = B_396;
        jcb(index,1433) = - B_194+ 0.98*B_196;
        jcb(index,1434) = - B_138+ B_140+ B_142+ B_144+ 0.82*B_168;
        jcb(index,1435) = B_433;
        jcb(index,1436) = - B_154- B_156+ B_164+ 0.82*B_169;
        jcb(index,1437) = B_8- B_24- B_26- B_28;
        jcb(index,1438) = B_16- B_22+ 0.13*B_135+ 0.26*B_181+ 0.32*B_227+ 0.25*B_255;
        jcb(index,1439) = - B_114+ B_116+ B_118+ 2*B_120+ B_145+ B_165+ 0.8*B_191+ 0.3*B_207+ 1.23*B_219+ 0.875*B_237+ B_265+ B_311 + B_360;
        jcb(index,1440) = B_17- B_30+ B_36+ B_113+ B_124+ B_131+ B_211+ 0.94*B_288+ B_393+ B_397+ 0.4*B_401;
        jcb(index,1441) = - B_282- B_284+ B_287+ B_307;
        jcb(index,1442) = 0;
        jcb(index,1443) = - B_337+ B_354;
        jcb(index,1444) = B_125+ B_126+ B_308+ B_355+ B_429;
        jcb(index,1445) = - B_20;
        jcb(index,1446) = - B_339+ B_361;
        jcb(index,1447) = - B_62+ B_86+ B_117+ B_141+ 0.96*B_189+ 0.98*B_197+ 0.56*B_223+ 0.75*B_233+ 0.956*B_263;
        jcb(index,1448) = 0.94*B_289- B_290+ B_312;
        jcb(index,1449) = - B_66;
        jcb(index,1450) = - B_68+ B_119+ B_127+ B_143;
        jcb(index,1451) = - B_21- B_23- B_25- B_27- B_29- B_31- 2*B_32- B_63- B_67- B_69- B_82- B_84- B_115- B_139- B_155- B_157 - B_187- B_195- B_203- B_221- B_231- B_244- B_261- B_283- B_285- B_291- B_338- B_340- B_406;
        jcb(index,1452) = 0;
        jcb(index,1453) = - B_490;
        jcb(index,1454) = B_286;
        jcb(index,1455) = B_280;
        jcb(index,1456) = B_305;
        jcb(index,1457) = B_323;
        jcb(index,1458) = B_309;
        jcb(index,1459) = - B_498;
        jcb(index,1460) = 0;
        jcb(index,1461) = 0;
        jcb(index,1462) = - B_484;
        jcb(index,1463) = B_407;
        jcb(index,1464) = - B_486;
        jcb(index,1465) = - B_502;
        jcb(index,1466) = B_325;
        jcb(index,1467) = 0;
        jcb(index,1468) = 0;
        jcb(index,1469) = 0;
        jcb(index,1470) = 0;
        jcb(index,1471) = 0;
        jcb(index,1472) = 0;
        jcb(index,1473) = 0.06*B_288- B_292;
        jcb(index,1474) = B_281+ B_282+ B_287+ B_306+ B_307+ B_310+ B_324+ B_326+ B_408;
        jcb(index,1475) = 0;
        jcb(index,1476) = 0;
        jcb(index,1477) = B_308;
        jcb(index,1478) = 0;
        jcb(index,1479) = 0;
        jcb(index,1480) = 0;
        jcb(index,1481) = 0.06*B_289;
        jcb(index,1482) = 0;
        jcb(index,1483) = 0;
        jcb(index,1484) = B_283;
        jcb(index,1485) = - B_293- B_456- B_485- B_487- B_491- B_499- B_503;
    }

__device__ void Fun(double *var, const double * __restrict__ fix, const double * __restrict__ rconst, double *varDot, int &Nfun, const int VL_GLO){
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    Nfun++;

 double dummy, A_0, A_1, A_2, A_3, A_4, A_5, A_6, A_7, A_8, A_9, A_10, A_11, A_12, A_13, A_14, A_15, A_16, A_17, A_18, A_19, A_20, A_21, A_22, A_23, A_24, A_25, A_26, A_27, A_28, A_29, A_30, A_31, A_32, A_33, A_34, A_35, A_36, A_37, A_38, A_39, A_40, A_41, A_42, A_43, A_44, A_45, A_46, A_47, A_48, A_49, A_50, A_51, A_52, A_53, A_54, A_55, A_56, A_57, A_58, A_59, A_60, A_61, A_62, A_63, A_64, A_65, A_66, A_67, A_68, A_69, A_70, A_71, A_72, A_73, A_74, A_75, A_76, A_77, A_78, A_79, A_80, A_81, A_82, A_83, A_84, A_85, A_86, A_87, A_88, A_89, A_90, A_91, A_92, A_93, A_94, A_95, A_96, A_97, A_98, A_99, A_100, A_101, A_102, A_103, A_104, A_105, A_106, A_107, A_108, A_109, A_110, A_111, A_112, A_113, A_114, A_115, A_116, A_117, A_118, A_119, A_120, A_121, A_122, A_123, A_124, A_125, A_126, A_127, A_128, A_129, A_130, A_131, A_132, A_133, A_134, A_135, A_136, A_137, A_138, A_139, A_140, A_141, A_142, A_143, A_144, A_145, A_146, A_147, A_148, A_149, A_150, A_151, A_152, A_153, A_154, A_155, A_156, A_157, A_158, A_159, A_160, A_161, A_162, A_163, A_164, A_165, A_166, A_167, A_168, A_169, A_170, A_171, A_172, A_173, A_174, A_175, A_176, A_177, A_178, A_179, A_180, A_181, A_182, A_183, A_184, A_185, A_186, A_187, A_188, A_189, A_190, A_191, A_192, A_193, A_194, A_195, A_196, A_197, A_198, A_199, A_200, A_201, A_202, A_203, A_204, A_205, A_206, A_207, A_208, A_209, A_210, A_211, A_212, A_213, A_214, A_215, A_216, A_217, A_218, A_219, A_220, A_221, A_222, A_223, A_224, A_225, A_226, A_227, A_228, A_229, A_230, A_231, A_232, A_233, A_234, A_235, A_236, A_237, A_238, A_239, A_240, A_241, A_242, A_243, A_244, A_245, A_246, A_247, A_248, A_249, A_250, A_251, A_252, A_253, A_254, A_255, A_256, A_257, A_258, A_259, A_260, A_261, A_262, A_263, A_264, A_265, A_266, A_267, A_268, A_269, A_270, A_271, A_272, A_273, A_274, A_275, A_276, A_277, A_278, A_279, A_280, A_281, A_282, A_283, A_284, A_285, A_286, A_287, A_288, A_289, A_290, A_291, A_292, A_293, A_294, A_295, A_296, A_297, A_298, A_299, A_300, A_301, A_302, A_303, A_304, A_305, A_306, A_307, A_308, A_309;

    {
        A_0 = rconst(index,0)*var(index,120)*fix(index,0);
        A_1 = rconst(index,1)*var(index,131)*fix(index,0);
        A_2 = 1.2e-10*var(index,120)*var(index,124);
        A_3 = rconst(index,3)*var(index,124)*var(index,131);
        A_4 = rconst(index,4)*var(index,122)*fix(index,0);
        A_5 = rconst(index,5)*var(index,122)*var(index,124);
        A_6 = 1.2e-10*var(index,97)*var(index,120);
        A_7 = rconst(index,7)*var(index,126)*var(index,131);
        A_8 = rconst(index,8)*var(index,124)*var(index,126);
        A_9 = rconst(index,9)*var(index,97)*var(index,126);
        A_10 = rconst(index,10)*var(index,131)*var(index,137);
        A_11 = rconst(index,11)*var(index,124)*var(index,137);
        A_12 = 7.2e-11*var(index,122)*var(index,137);
        A_13 = 6.9e-12*var(index,122)*var(index,137);
        A_14 = 1.6e-12*var(index,122)*var(index,137);
        A_15 = rconst(index,15)*var(index,126)*var(index,137);
        A_16 = rconst(index,16)*var(index,137)*var(index,137);
        A_17 = rconst(index,17)*var(index,120)*var(index,128);
        A_18 = 1.8e-12*var(index,88)*var(index,126);
        A_19 = rconst(index,19)*var(index,59)*fix(index,0);
        A_20 = rconst(index,20)*var(index,120)*fix(index,1);
        A_21 = rconst(index,21)*var(index,60)*var(index,120);
        A_22 = rconst(index,22)*var(index,60)*var(index,120);
        A_23 = rconst(index,23)*var(index,124)*var(index,133);
        A_24 = rconst(index,24)*var(index,59)*var(index,133);
        A_25 = rconst(index,25)*var(index,131)*var(index,135);
        A_26 = rconst(index,26)*var(index,124)*var(index,135);
        A_27 = rconst(index,27)*var(index,59)*var(index,135);
        A_28 = rconst(index,28)*var(index,133)*var(index,136);
        A_29 = rconst(index,29)*var(index,135)*var(index,136);
        A_30 = rconst(index,30)*var(index,83);
        A_31 = rconst(index,31)*var(index,126)*var(index,133);
        A_32 = rconst(index,32)*var(index,133)*var(index,137);
        A_33 = rconst(index,33)*var(index,126)*var(index,135);
        A_34 = rconst(index,34)*var(index,135)*var(index,137);
        A_35 = 3.5e-12*var(index,136)*var(index,137);
        A_36 = rconst(index,36)*var(index,76)*var(index,126);
        A_37 = rconst(index,37)*var(index,101)*var(index,126);
        A_38 = rconst(index,38)*var(index,73);
        A_39 = rconst(index,39)*var(index,73)*var(index,126);
        A_40 = rconst(index,40)*var(index,47)*var(index,126);
        A_41 = rconst(index,41)*var(index,92)*var(index,124);
        A_42 = rconst(index,42)*var(index,92)*var(index,137);
        A_43 = rconst(index,43)*var(index,92)*var(index,137);
        A_44 = rconst(index,44)*var(index,92)*var(index,133);
        A_45 = rconst(index,45)*var(index,92)*var(index,133);
        A_46 = rconst(index,46)*var(index,92)*var(index,135);
        A_47 = rconst(index,47)*var(index,92)*var(index,135);
        A_48 = 1.2e-14*var(index,84)*var(index,124);
        A_49 = 1300*var(index,84);
        A_50 = rconst(index,50)*var(index,87)*var(index,126);
        A_51 = rconst(index,51)*var(index,70)*var(index,87);
        A_52 = rconst(index,52)*var(index,87)*var(index,135);
        A_53 = 1.66e-12*var(index,70)*var(index,126);
        A_54 = rconst(index,54)*var(index,61)*var(index,126);
        A_55 = rconst(index,55)*var(index,87)*fix(index,0);
        A_56 = 1.75e-10*var(index,98)*var(index,120);
        A_57 = rconst(index,57)*var(index,98)*var(index,126);
        A_58 = rconst(index,58)*var(index,89)*var(index,126);
        A_59 = rconst(index,59)*var(index,125)*var(index,137);
        A_60 = rconst(index,60)*var(index,125)*var(index,133);
        A_61 = 1.3e-12*var(index,125)*var(index,136);
        A_62 = rconst(index,62)*var(index,125)*var(index,125);
        A_63 = rconst(index,63)*var(index,125)*var(index,125);
        A_64 = rconst(index,64)*var(index,104)*var(index,126);
        A_65 = rconst(index,65)*var(index,126)*var(index,130);
        A_66 = rconst(index,66)*var(index,130)*var(index,136);
        A_67 = rconst(index,67)*var(index,95)*var(index,126);
        A_68 = 4e-13*var(index,78)*var(index,126);
        A_69 = rconst(index,69)*var(index,48)*var(index,126);
        A_70 = rconst(index,70)*var(index,103)*var(index,124);
        A_71 = rconst(index,71)*var(index,103)*var(index,126);
        A_72 = rconst(index,72)*var(index,117)*var(index,137);
        A_73 = rconst(index,73)*var(index,117)*var(index,133);
        A_74 = 2.3e-12*var(index,117)*var(index,136);
        A_75 = rconst(index,75)*var(index,117)*var(index,125);
        A_76 = rconst(index,76)*var(index,71)*var(index,126);
        A_77 = rconst(index,77)*var(index,119)*var(index,126);
        A_78 = rconst(index,78)*var(index,119)*var(index,136);
        A_79 = rconst(index,79)*var(index,74)*var(index,126);
        A_80 = rconst(index,80)*var(index,121)*var(index,137);
        A_81 = rconst(index,81)*var(index,121)*var(index,137);
        A_82 = rconst(index,82)*var(index,121)*var(index,133);
        A_83 = rconst(index,83)*var(index,121)*var(index,135);
        A_84 = 4e-12*var(index,121)*var(index,136);
        A_85 = rconst(index,85)*var(index,121)*var(index,125);
        A_86 = rconst(index,86)*var(index,121)*var(index,125);
        A_87 = rconst(index,87)*var(index,117)*var(index,121);
        A_88 = rconst(index,88)*var(index,121)*var(index,121);
        A_89 = rconst(index,89)*var(index,63)*var(index,126);
        A_90 = rconst(index,90)*var(index,58)*var(index,126);
        A_91 = rconst(index,91)*var(index,77)*var(index,126);
        A_92 = rconst(index,92)*var(index,77);
        A_93 = rconst(index,93)*var(index,49)*var(index,126);
        A_94 = rconst(index,94)*var(index,107)*var(index,124);
        A_95 = rconst(index,95)*var(index,107)*var(index,126);
        A_96 = rconst(index,96)*var(index,107)*var(index,136);
        A_97 = rconst(index,97)*var(index,93)*var(index,137);
        A_98 = rconst(index,98)*var(index,93)*var(index,133);
        A_99 = rconst(index,99)*var(index,93)*var(index,125);
        A_100 = rconst(index,100)*var(index,69)*var(index,126);
        A_101 = rconst(index,101)*var(index,115)*var(index,137);
        A_102 = rconst(index,102)*var(index,115)*var(index,133);
        A_103 = rconst(index,103)*var(index,67)*var(index,126);
        A_104 = rconst(index,104)*var(index,86)*var(index,126);
        A_105 = rconst(index,105)*var(index,94)*var(index,137);
        A_106 = rconst(index,106)*var(index,94)*var(index,133);
        A_107 = rconst(index,107)*var(index,94)*var(index,125);
        A_108 = rconst(index,108)*var(index,72)*var(index,126);
        A_109 = rconst(index,109)*var(index,108)*var(index,126);
        A_110 = rconst(index,110)*var(index,96)*var(index,126);
        A_111 = rconst(index,111)*var(index,62)*var(index,126);
        A_112 = rconst(index,112)*var(index,40)*var(index,126);
        A_113 = rconst(index,113)*var(index,102)*var(index,125);
        A_114 = rconst(index,114)*var(index,102)*var(index,137);
        A_115 = rconst(index,115)*var(index,102)*var(index,133);
        A_116 = rconst(index,116)*var(index,79)*var(index,126);
        A_117 = rconst(index,117)*var(index,110)*var(index,124);
        A_118 = rconst(index,118)*var(index,110)*var(index,126);
        A_119 = rconst(index,119)*var(index,113)*var(index,137);
        A_120 = rconst(index,120)*var(index,113)*var(index,133);
        A_121 = rconst(index,121)*var(index,113)*var(index,135);
        A_122 = 2e-12*var(index,113)*var(index,125);
        A_123 = 2e-12*var(index,113)*var(index,113);
        A_124 = 3e-11*var(index,82)*var(index,126);
        A_125 = rconst(index,125)*var(index,85)*var(index,126);
        A_126 = rconst(index,126)*var(index,99)*var(index,137);
        A_127 = rconst(index,127)*var(index,99)*var(index,133);
        A_128 = rconst(index,128)*var(index,68)*var(index,126);
        A_129 = 1.7e-12*var(index,111)*var(index,126);
        A_130 = 3.2e-11*var(index,64)*var(index,126);
        A_131 = rconst(index,131)*var(index,64);
        A_132 = rconst(index,132)*var(index,106)*var(index,124);
        A_133 = rconst(index,133)*var(index,106)*var(index,126);
        A_134 = rconst(index,134)*var(index,106)*var(index,136);
        A_135 = rconst(index,135)*var(index,109)*var(index,137);
        A_136 = rconst(index,136)*var(index,109)*var(index,133);
        A_137 = 2e-12*var(index,109)*var(index,125);
        A_138 = 2e-12*var(index,109)*var(index,109);
        A_139 = 1e-10*var(index,66)*var(index,126);
        A_140 = 1.3e-11*var(index,91)*var(index,126);
        A_141 = rconst(index,141)*var(index,124)*var(index,127);
        A_142 = rconst(index,142)*var(index,131)*var(index,134);
        A_143 = rconst(index,143)*var(index,134)*var(index,134);
        A_144 = rconst(index,144)*var(index,134)*var(index,134);
        A_145 = rconst(index,145)*var(index,134)*var(index,134);
        A_146 = rconst(index,146)*var(index,134)*var(index,134);
        A_147 = rconst(index,147)*var(index,39);
        A_148 = rconst(index,148)*var(index,97)*var(index,127);
        A_149 = rconst(index,149)*var(index,127)*var(index,137);
        A_150 = rconst(index,150)*var(index,127)*var(index,137);
        A_151 = rconst(index,151)*var(index,88)*var(index,127);
        A_152 = rconst(index,152)*var(index,126)*var(index,134);
        A_153 = rconst(index,153)*var(index,134)*var(index,137);
        A_154 = rconst(index,154)*var(index,126)*var(index,138);
        A_155 = rconst(index,155)*var(index,112)*var(index,126);
        A_156 = rconst(index,156)*var(index,133)*var(index,134);
        A_157 = rconst(index,157)*var(index,134)*var(index,135);
        A_158 = rconst(index,158)*var(index,116);
        A_159 = rconst(index,159)*var(index,116)*var(index,131);
        A_160 = rconst(index,160)*var(index,116)*var(index,127);
        A_161 = rconst(index,161)*var(index,98)*var(index,127);
        A_162 = rconst(index,162)*var(index,127)*var(index,130);
        A_163 = 5.9e-11*var(index,104)*var(index,127);
        A_164 = rconst(index,164)*var(index,125)*var(index,134);
        A_165 = 3.3e-10*var(index,41)*var(index,120);
        A_166 = 1.65e-10*var(index,75)*var(index,120);
        A_167 = rconst(index,167)*var(index,75)*var(index,126);
        A_168 = 3.25e-10*var(index,57)*var(index,120);
        A_169 = rconst(index,169)*var(index,57)*var(index,126);
        A_170 = rconst(index,170)*var(index,103)*var(index,127);
        A_171 = 8e-11*var(index,119)*var(index,127);
        A_172 = 1.4e-10*var(index,42)*var(index,120);
        A_173 = 2.3e-10*var(index,43)*var(index,120);
        A_174 = rconst(index,174)*var(index,124)*var(index,129);
        A_175 = rconst(index,175)*var(index,131)*var(index,132);
        A_176 = 2.7e-12*var(index,132)*var(index,132);
        A_177 = rconst(index,177)*var(index,132)*var(index,132);
        A_178 = rconst(index,178)*var(index,129)*var(index,137);
        A_179 = rconst(index,179)*var(index,132)*var(index,137);
        A_180 = rconst(index,180)*var(index,123)*var(index,126);
        A_181 = rconst(index,181)*var(index,118)*var(index,131);
        A_182 = rconst(index,182)*var(index,100)*var(index,126);
        A_183 = 4.9e-11*var(index,105)*var(index,129);
        A_184 = rconst(index,184)*var(index,132)*var(index,133);
        A_185 = rconst(index,185)*var(index,132)*var(index,135);
        A_186 = rconst(index,186)*var(index,105);
        A_187 = rconst(index,187)*var(index,129)*var(index,130);
        A_188 = rconst(index,188)*var(index,104)*var(index,129);
        A_189 = rconst(index,189)*var(index,125)*var(index,132);
        A_190 = rconst(index,190)*var(index,125)*var(index,132);
        A_191 = rconst(index,191)*var(index,53)*var(index,126);
        A_192 = rconst(index,192)*var(index,103)*var(index,129);
        A_193 = rconst(index,193)*var(index,119)*var(index,129);
        A_194 = rconst(index,194)*var(index,45)*var(index,126);
        A_195 = rconst(index,195)*var(index,44)*var(index,126);
        A_196 = 3.32e-15*var(index,90)*var(index,129);
        A_197 = 1.1e-15*var(index,80)*var(index,129);
        A_198 = rconst(index,198)*var(index,100)*var(index,127);
        A_199 = rconst(index,199)*var(index,132)*var(index,134);
        A_200 = rconst(index,200)*var(index,132)*var(index,134);
        A_201 = rconst(index,201)*var(index,132)*var(index,134);
        A_202 = 1.45e-11*var(index,90)*var(index,127);
        A_203 = rconst(index,203)*var(index,54)*var(index,126);
        A_204 = rconst(index,204)*var(index,55)*var(index,126);
        A_205 = rconst(index,205)*var(index,52)*var(index,126);
        A_206 = rconst(index,206)*var(index,56)*var(index,126);
        A_207 = rconst(index,207)*var(index,114)*var(index,126);
        A_208 = rconst(index,208)*var(index,114)*var(index,126);
        A_209 = rconst(index,209)*var(index,114)*var(index,136);
        A_210 = 1e-10*var(index,65)*var(index,126);
        A_211 = rconst(index,211)*var(index,81);
        A_212 = 3e-13*var(index,81)*var(index,124);
        A_213 = 5e-11*var(index,46)*var(index,137);
        A_214 = 3.3e-10*var(index,114)*var(index,127);
        A_215 = rconst(index,215)*var(index,114)*var(index,129);
        A_216 = 4.4e-13*var(index,114)*var(index,132);
        A_217 = rconst(index,217)*fix(index,0);
        A_218 = rconst(index,218)*var(index,124);
        A_219 = rconst(index,219)*var(index,124);
        A_220 = rconst(index,220)*var(index,128);
        A_221 = rconst(index,221)*var(index,88);
        A_222 = rconst(index,222)*var(index,60);
        A_223 = rconst(index,223)*var(index,135);
        A_224 = rconst(index,224)*var(index,133);
        A_225 = rconst(index,225)*var(index,136);
        A_226 = rconst(index,226)*var(index,136);
        A_227 = rconst(index,227)*var(index,83);
        A_228 = rconst(index,228)*var(index,76);
        A_229 = rconst(index,229)*var(index,101);
        A_230 = rconst(index,230)*var(index,73);
        A_231 = rconst(index,231)*var(index,104);
        A_232 = rconst(index,232)*var(index,130);
        A_233 = rconst(index,233)*var(index,130);
        A_234 = rconst(index,234)*fix(index,2);
        A_235 = rconst(index,235)*var(index,98);
        A_236 = rconst(index,236)*var(index,71);
        A_237 = rconst(index,237)*var(index,119);
        A_238 = rconst(index,238)*var(index,63);
        A_239 = rconst(index,239)*var(index,58);
        A_240 = rconst(index,240)*var(index,77);
        A_241 = rconst(index,241)*var(index,69);
        A_242 = rconst(index,242)*var(index,86);
        A_243 = rconst(index,243)*var(index,108);
        A_244 = rconst(index,244)*var(index,96);
        A_245 = rconst(index,245)*var(index,72);
        A_246 = rconst(index,246)*var(index,62);
        A_247 = rconst(index,247)*var(index,79);
        A_248 = rconst(index,248)*var(index,110);
        A_249 = rconst(index,249)*var(index,82);
        A_250 = rconst(index,250)*var(index,85);
        A_251 = rconst(index,251)*var(index,68);
        A_252 = rconst(index,252)*var(index,38);
        A_253 = rconst(index,253)*var(index,111);
        A_254 = rconst(index,254)*var(index,64);
        A_255 = rconst(index,255)*var(index,66);
        A_256 = rconst(index,256)*var(index,91);
        A_257 = rconst(index,257)*var(index,80);
        A_258 = rconst(index,258)*var(index,39);
        A_259 = rconst(index,259)*var(index,51);
        A_260 = rconst(index,260)*var(index,138);
        A_261 = rconst(index,261)*var(index,112);
        A_262 = rconst(index,262)*var(index,50);
        A_263 = rconst(index,263)*var(index,116);
        A_264 = rconst(index,264)*var(index,116);
        A_265 = rconst(index,265)*var(index,75);
        A_266 = rconst(index,266)*var(index,41);
        A_267 = rconst(index,267)*var(index,57);
        A_268 = rconst(index,268)*var(index,43);
        A_269 = rconst(index,269)*var(index,42);
        A_270 = rconst(index,270)*var(index,100);
        A_271 = rconst(index,271)*var(index,132);
        A_272 = rconst(index,272)*var(index,118);
        A_273 = rconst(index,273)*var(index,0);
        A_274 = rconst(index,274)*var(index,105);
        A_275 = rconst(index,275)*var(index,53);
        A_276 = rconst(index,276)*var(index,44);
        A_277 = rconst(index,277)*var(index,45);
        A_278 = rconst(index,278)*var(index,2);
        A_279 = rconst(index,279)*var(index,90);
        A_280 = rconst(index,280)*var(index,1);
        A_281 = rconst(index,281)*var(index,52);
        A_282 = rconst(index,282)*var(index,54);
        A_283 = rconst(index,283)*var(index,55);
        A_284 = rconst(index,284)*var(index,3);
        A_285 = rconst(index,285)*var(index,83)*var(index,128);
        A_286 = rconst(index,286)*var(index,83);
        A_287 = rconst(index,287)*var(index,112)*var(index,138);
        A_288 = rconst(index,288)*var(index,116)*var(index,138);
        A_289 = rconst(index,289)*var(index,116)*var(index,128);
        A_290 = rconst(index,290)*var(index,83)*var(index,138);
        A_291 = rconst(index,291)*var(index,118)*var(index,123);
        A_292 = rconst(index,292)*var(index,105)*var(index,128);
        A_293 = rconst(index,293)*var(index,116)*var(index,123);
        A_294 = rconst(index,294)*var(index,105)*var(index,138);
        A_295 = rconst(index,295)*var(index,112)*var(index,123);
        A_296 = rconst(index,296)*var(index,118)*var(index,138);
        A_297 = rconst(index,297)*var(index,4);
        A_298 = 2.3e-10*var(index,15)*var(index,120);
        A_299 = rconst(index,299)*var(index,15);
        A_300 = 1.4e-10*var(index,16)*var(index,120);
        A_301 = rconst(index,301)*var(index,16);
        A_302 = rconst(index,302)*var(index,17)*var(index,120);
        A_303 = rconst(index,303)*var(index,17)*var(index,120);
        A_304 = rconst(index,304)*var(index,17);
        A_305 = 3e-10*var(index,18)*var(index,120);
        A_306 = rconst(index,306)*var(index,18)*var(index,126);
        A_307 = rconst(index,307)*var(index,18);
        A_308 = rconst(index,308)*var(index,5);
        A_309 = rconst(index,309)*var(index,6);
        varDot(index,0) = - A_273;
        varDot(index,1) = - A_280;
        varDot(index,2) = - A_278;
        varDot(index,3) = - A_284;
        varDot(index,4) = - A_297;
        varDot(index,5) = - A_308;
        varDot(index,6) = - A_309;
        varDot(index,7) = A_165+ 0.9*A_166+ A_167+ 2*A_168+ 2*A_169+ A_172+ A_173+ A_191+ A_194+ A_195+ A_203+ A_204+ A_205+ A_266+ 2 *A_267+ A_268+ A_269+ A_276+ A_277+ A_278+ A_280+ A_281+ A_282+ A_283;
        varDot(index,8) = 2*A_172+ A_173+ A_268+ 2*A_269+ 3*A_278+ 2*A_280;
        varDot(index,9) = 0.09*A_166+ 2*A_203+ A_204+ A_205+ 2*A_268+ A_269;
        varDot(index,10) = 0.4*A_210+ A_213;
        varDot(index,11) = A_206;
        varDot(index,12) = 2*A_286;
        varDot(index,13) = 2*A_286;
        varDot(index,14) = A_299+ A_301+ A_303+ A_304+ A_307+ A_308+ A_309;
        varDot(index,15) = - A_298- A_299;
        varDot(index,16) = - A_300- A_301;
        varDot(index,17) = - A_302- A_303- A_304;
        varDot(index,18) = - A_305- A_306- A_307;
        varDot(index,19) = A_297;
        varDot(index,20) = A_11;
        varDot(index,21) = A_17;
        varDot(index,22) = 2*A_2+ 2*A_3+ A_5+ A_6+ A_7+ A_8+ A_10+ A_11+ A_17+ A_21+ A_22+ 2*A_25+ A_35+ A_41+ A_46+ A_47+ A_48+ A_52 + A_56+ A_61+ A_66+ A_70+ A_74+ A_78+ A_84+ A_94+ A_96+ A_117+ A_132+ A_134+ 2*A_142+ 2*A_143+ 2*A_144 + A_145+ A_152+ A_164+ A_166+ A_168+ 2*A_175+ 2*A_176+ 2*A_177+ A_181+ A_190+ A_199+ 2*A_200+ 2*A_201+ 2 *A_226+ 2*A_258+ A_261+ A_272+ A_285+ 3*A_286+ A_287+ A_288+ 2*A_290+ A_291+ A_293+ A_294+ A_295+ A_296;
        varDot(index,23) = 2*A_175+ 2*A_176+ 2*A_177+ A_181+ A_190+ 0.5*A_199+ A_200+ A_201+ A_272+ A_291+ 0.333333*A_293+ 0.333333  *A_294+ 0.5*A_295+ 0.5*A_296;
        varDot(index,24) = 2*A_142+ 2*A_143+ 2*A_144+ A_145+ A_152+ A_164+ A_166+ A_168+ 0.5*A_199+ A_200+ A_201+ 2*A_258+ A_261  + A_287+ 0.5*A_288+ A_290+ 0.333333*A_293+ 0.333333*A_294+ 0.5*A_295+ 0.5*A_296;
        varDot(index,25) = A_5+ A_6+ A_7+ A_8+ A_10+ A_11;
        varDot(index,26) = 2*A_25+ A_35+ A_41+ A_46+ A_47+ A_48+ A_52+ 2*A_226+ A_285+ 3*A_286+ 0.5*A_288+ A_290+ 0.333333*A_293  + 0.333333*A_294;
        varDot(index,27) = 2*A_2+ 2*A_3+ A_17+ A_21+ A_22+ A_56;
        varDot(index,28) = A_61+ A_66+ A_70+ A_74+ A_78+ A_84+ A_94+ A_96+ A_117+ A_132+ A_134;
        varDot(index,29) = A_8;
        varDot(index,30) = A_32;
        varDot(index,31) = A_191+ A_275+ A_278+ A_280;
        varDot(index,32) = 4*A_165+ A_166+ A_167+ 3*A_168+ 3*A_169+ 2*A_172+ 3*A_173+ A_265+ 4*A_266+ 3*A_267+ 3*A_268+ 2*A_269  + A_280;
        varDot(index,33) = A_60;
        varDot(index,34) = A_14+ A_19+ A_24+ A_32+ A_36+ A_37+ A_60+ A_73+ A_81+ A_82+ A_98+ A_102+ A_106+ A_115+ A_120+ A_127+ A_136 + A_150+ A_182+ A_207+ A_208+ 0.4*A_210+ A_214+ A_215+ 2*A_217+ A_222+ A_224+ 0.333*A_230+ A_234+ A_259 + A_262+ A_273;
        varDot(index,35) = A_73+ A_82+ A_98+ A_102+ A_106+ A_115+ A_120+ A_127+ A_136;
        varDot(index,36) = 3*A_194+ 2*A_195+ A_203+ 2*A_204+ A_205+ 2*A_276+ 3*A_277+ A_281+ A_282+ 2*A_283;
        varDot(index,37) = A_281+ 2*A_282+ A_283;
        varDot(index,38) = 0.8*A_128- A_252;
        varDot(index,39) = A_146- A_147- A_258;
        varDot(index,40) = - A_112;
        varDot(index,41) = - A_165- A_266;
        varDot(index,42) = - A_172- A_269;
        varDot(index,43) = - A_173- A_268;
        varDot(index,44) = - A_195- A_276;
        varDot(index,45) = - A_194- A_277;
        varDot(index,46) = A_212- A_213;
        varDot(index,47) = - A_40;
        varDot(index,48) = - A_69;
        varDot(index,49) = - A_93;
        varDot(index,50) = - A_262+ A_290;
        varDot(index,51) = A_145+ A_199- A_259;
        varDot(index,52) = - A_205- A_281;
        varDot(index,53) = - A_191- A_275;
        varDot(index,54) = - A_203- A_282;
        varDot(index,55) = - A_204- A_283;
        varDot(index,56) = - A_206+ 0.6*A_210+ A_211;
        varDot(index,57) = - A_168- A_169- A_267;
        varDot(index,58) = - A_90+ A_140- A_239;
        varDot(index,59) = - A_19- A_24- A_27+ A_224;
        varDot(index,60) = - A_21- A_22+ A_27+ A_46- A_222;
        varDot(index,61) = A_51- A_54;
        varDot(index,62) = 0.04*A_98- A_111- A_246;
        varDot(index,63) = A_80- A_89- A_238;
        varDot(index,64) = A_121- A_130- A_131- A_254;
        varDot(index,65) = A_208- A_210+ A_216;
        varDot(index,66) = A_135- A_139- A_255;
        varDot(index,67) = A_101- A_103;
        varDot(index,68) = A_126- A_128- A_251;
        varDot(index,69) = A_97- A_100- A_241;
        varDot(index,70) = A_49- A_51- A_53+ A_54;
        varDot(index,71) = A_72- A_76- A_236;
        varDot(index,72) = A_105- A_108- A_245;
        varDot(index,73) = A_34- A_38- A_39- A_230;
        varDot(index,74) = - A_79+ A_81+ A_86+ 0.18*A_87;
        varDot(index,75) = - 0.9*A_166- A_167- A_265;
        varDot(index,76) = A_31- A_36+ A_52- A_228;
        varDot(index,77) = A_83- A_91- A_92- A_240;
        varDot(index,78) = - A_68+ 0.23125*A_70+ 0.22*A_94+ 0.45*A_117+ 0.28*A_132;
        varDot(index,79) = A_114- A_116- A_247;
        varDot(index,80) = A_143+ A_160- A_197+ A_202- A_257+ A_287+ A_288;
        varDot(index,81) = A_207+ A_209- A_211- A_212+ A_214+ A_215;
        varDot(index,82) = A_119- A_124- A_249;
        varDot(index,83) = A_29- A_30- A_227- A_285- A_286- A_290;
        varDot(index,84) = A_41+ A_42+ A_47- A_48- A_49;
        varDot(index,85) = 0.88*A_113+ 0.56*A_115+ 0.85*A_116- A_125+ A_129+ 0.67*A_247- A_250+ 0.67*A_253;
        varDot(index,86) = 0.96*A_98+ A_99+ 0.7*A_100- A_104+ A_111+ A_241- A_242+ A_246;
        varDot(index,87) = A_43- A_50- A_51- A_52+ A_53- A_55;
        varDot(index,88) = A_16- A_18+ 0.13875*A_70+ 0.09*A_132- A_151- A_221;
        varDot(index,89) = - A_58+ A_63+ 0.25*A_75+ 0.03*A_94+ 0.2*A_99+ 0.5*A_107+ 0.18*A_113+ 0.25*A_122+ 0.25*A_137;
        varDot(index,90) = - A_196+ A_197+ A_198+ A_201- A_202- A_279+ A_293+ A_294+ A_295+ A_296;
        varDot(index,91) = A_134+ 0.044*A_136- A_140- A_256;
        varDot(index,92) = A_40- A_41- A_42- A_43- A_44- A_45- A_46- A_47+ A_48;
        varDot(index,93) = 0.82*A_93- A_97- A_98- A_99+ 0.3*A_100;
        varDot(index,94) = A_104- A_105- A_106- A_107+ 0.3*A_108;
        varDot(index,95) = A_65+ A_66- A_67+ 0.63*A_70+ A_90+ A_91+ 0.31*A_94+ A_110+ 0.22*A_117+ 0.25*A_120+ 0.125*A_122+ 0.5*A_123 + 0.14*A_132+ A_162+ A_187+ A_232+ A_233+ A_234+ A_235+ A_237+ A_239+ A_244+ A_248+ 0.25*A_249;
        varDot(index,96) = 0.04*A_94+ 0.5*A_107+ 0.7*A_108+ A_109- A_110+ 0.9*A_117+ 0.5*A_120+ 0.5*A_122+ A_123+ 0.25*A_137- A_244  + 0.5*A_249;
        varDot(index,97) = - A_6- A_9+ A_13+ 0.05*A_56- A_148+ A_232+ 0.69*A_235;
        varDot(index,98) = - A_56- A_57+ 0.06*A_94- A_161- A_235;
        varDot(index,99) = A_125- A_126- A_127+ 0.2*A_128;
        varDot(index,100) = A_177- A_182+ A_183+ A_196- A_198- A_270+ A_291;
        varDot(index,101) = A_33- A_37+ A_66+ A_78+ A_209- A_229+ 2*A_285+ A_288+ A_289+ A_290+ A_292+ A_293+ A_294;
        varDot(index,102) = A_112- A_113- A_114- A_115+ 0.15*A_116;
        varDot(index,103) = - A_70- A_71- A_170- A_192;
        varDot(index,104) = A_59- A_64- A_163- A_188- A_231;
        varDot(index,105) = - A_183+ A_185- A_186- A_274- A_292- A_294;
        varDot(index,106) = - A_132- A_133- A_134;
        varDot(index,107) = - A_94- A_95- A_96;
        varDot(index,108) = 0.5*A_103+ 0.2*A_107- A_109+ 0.25*A_120+ 0.375*A_122+ A_123+ A_130+ 0.25*A_137+ A_140- A_243+ 0.25*A_249 + A_254;
        varDot(index,109) = A_133- A_135- A_136- A_137- 2*A_138;
        varDot(index,110) = - A_117- A_118+ 0.65*A_132+ 0.956*A_136+ 0.5*A_137+ 2*A_138+ A_139- A_248+ A_255+ A_256;
        varDot(index,111) = A_96+ 0.02*A_102+ 0.16*A_115+ 0.015*A_127- A_129- A_253;
        varDot(index,112) = A_153- A_155- A_261- A_287+ A_289- A_295;
        varDot(index,113) = A_118- A_119- A_120- A_121- A_122- 2*A_123+ A_124+ A_131+ 0.1*A_132;
        varDot(index,114) = - A_207- A_208- A_209- A_214- A_215- A_216;
        varDot(index,115) = 0.666667*A_71+ A_95- A_101- A_102+ 0.5*A_103+ 0.666667*A_170+ 0.666667*A_192;
        varDot(index,116) = A_157- A_158- A_159- A_160- A_263- A_264- A_288- A_289- A_293;
        varDot(index,117) = A_69- A_72- A_73- A_74- A_75+ 0.3*A_76- A_87+ 0.18*A_93+ 0.06*A_94+ 0.12*A_113+ 0.28*A_115+ 0.33*A_247  + A_250+ 0.33*A_253;
        varDot(index,118) = A_179- A_181+ A_182+ A_189- A_272- A_291+ A_292- A_296;
        varDot(index,119) = A_73+ A_74+ 0.75*A_75+ 0.7*A_76- A_77- A_78+ A_87+ 0.47*A_94+ 0.98*A_102+ 0.12*A_113+ 0.28*A_115+ 0.985  *A_127- A_171- A_193+ A_236- A_237+ 0.33*A_247+ A_251+ 0.33*A_253;
        varDot(index,120) = - A_0- A_2- A_6- A_17- A_20- A_21- A_22- A_56- A_165- A_166- A_168- A_172- A_173+ A_218+ A_222;
        varDot(index,121) = A_77+ A_78- A_80- A_81- A_82- A_83- A_84- A_85- A_86- A_87- 2*A_88+ A_89+ A_92+ 0.23*A_94+ A_106+ 0.3 *A_107+ A_110+ 0.1*A_117+ 0.25*A_120+ 0.125*A_122+ 0.985*A_127+ 0.1*A_132+ A_171+ A_193+ A_240+ A_242 + A_243+ A_244+ A_245+ A_248+ 0.25*A_249+ A_250+ A_251+ 2*A_252;
        varDot(index,122) = - A_4- A_5+ A_6+ A_7+ A_9- A_12- A_13- A_14+ 0.4*A_56+ A_67+ A_148+ 0.09*A_166+ A_220+ A_233+ 0.31*A_235 + A_260;
        varDot(index,123) = A_178- A_180+ A_187+ A_188+ A_192+ A_193+ A_215- A_291- A_293- A_295;
        varDot(index,124) = A_1- A_2- A_3- A_5- A_8- A_11- A_23- A_26- A_41- A_48- A_70+ A_81- A_94- A_117- A_132- A_141- A_174 - A_212- A_218- A_219;
        varDot(index,125) = 0.75*A_56+ A_57- A_59- A_60- A_61- 2*A_62- 2*A_63+ 0.7*A_64- A_75+ A_79+ A_82+ A_84- A_86+ 0.82*A_87+ 2 *A_88+ 0.07*A_94- A_99- A_107- A_113- A_122+ 0.08*A_132- A_137+ A_161- A_164+ A_188- A_189- A_190+ 0.6 *A_210+ A_211+ A_237+ A_238+ A_242+ A_265+ A_275+ A_284;
        varDot(index,126) = A_5+ A_6- A_7- A_8- A_9+ A_10+ A_11+ 2*A_12- A_15+ 2*A_17- A_18- A_31+ A_32- A_33+ A_35- A_36- A_37 - A_39- A_40+ A_42+ A_44- A_50- A_53- A_54+ 0.75*A_56- A_57- A_58- 0.7*A_64- A_65- A_67- A_68- A_69+ 0.13 *A_70- A_71- 0.3*A_76- A_77- A_79- A_89- A_90- A_91- A_93+ 0.33*A_94- A_95- 0.3*A_100- 0.5*A_103- A_104 - 0.3*A_108- A_109- A_110- A_111- A_112- 0.15*A_116+ 0.19*A_117- A_118- A_124- A_125- 0.2*A_128- A_129 - A_130+ 0.25*A_132- A_133- A_140+ A_150- A_152- A_154- A_155+ A_163- A_167+ A_168- A_169- A_180+ A_181 - A_182- A_191- A_194- A_195- A_203- A_204- A_205- A_206- A_207- A_208- A_210+ A_220+ 2*A_221+ A_228+ A_229 + 0.333*A_230+ A_231+ A_236+ A_238+ A_241+ A_245+ A_247+ A_249+ A_251+ A_255+ A_261+ A_272;
        varDot(index,127) = - A_141+ A_142+ 2*A_144+ A_145- A_148- A_149- A_150- A_151+ 0.94*A_152+ A_154+ A_156- A_160- A_161- A_162 - A_163+ A_164+ 3*A_165+ 0.35*A_166+ A_167+ 3*A_168+ 3*A_169- A_170- A_171+ A_172+ 2*A_173+ A_196+ A_197 - A_198+ A_200- A_202- A_214+ 2*A_257+ 2*A_258+ A_260+ A_261+ A_262+ A_263+ A_265+ 4*A_266+ 3*A_267+ A_268 + A_269+ A_279+ A_280+ A_281+ 2*A_282+ A_283;
        varDot(index,128) = A_9+ A_14+ A_15- A_17+ A_18+ A_36+ A_37+ A_39+ A_40+ A_43+ A_45+ A_46+ A_50+ A_53+ A_54+ A_57+ A_64 + A_65+ A_68+ A_69+ A_77+ A_79+ A_89+ A_91+ A_93+ A_103+ A_104+ A_112+ 0.85*A_116+ A_129+ A_154+ A_155 + A_167+ A_169+ A_180+ A_191+ A_194+ A_195+ A_203+ A_204+ A_205- A_220+ 1.155*A_235- A_285+ A_287- A_289 + A_291- A_292+ A_295+ A_296;
        varDot(index,129) = - A_174+ A_175+ 2*A_176- A_178+ A_180+ A_182- A_183+ A_184- A_187- A_188+ A_190+ A_191- A_192- A_193+ 3 *A_194+ 2*A_195- A_196- A_197+ A_198+ A_199+ A_200+ A_202+ A_203+ 2*A_204+ A_205- A_215+ A_216+ 2*A_270 + A_271+ A_272+ A_273+ 0.85*A_274+ A_275+ 2*A_276+ 3*A_277+ A_278+ A_279+ A_280+ A_281+ A_282+ 2*A_283;
        varDot(index,130) = 0.25*A_56+ A_58+ A_60+ A_61+ 2*A_62+ A_63+ 0.3*A_64- A_65- A_66+ 1.13875*A_70+ 0.75*A_75+ A_85+ A_86  + A_90+ A_91+ 0.57*A_94+ 0.8*A_99+ 0.98*A_102+ A_106+ 0.8*A_107+ 0.68*A_113+ 0.75*A_120+ 1.125*A_122+ 0.5 *A_123+ 0.58*A_132+ 0.956*A_136+ 1.25*A_137+ A_138- A_162+ A_163+ A_164- A_187+ A_189+ A_190+ A_207+ A_209 + A_210+ A_214+ A_215+ A_231- A_232- A_233+ A_239+ A_243+ A_245+ A_248+ 0.75*A_249+ A_255+ A_256;
        varDot(index,131) = A_0- A_1- A_3- A_7- A_10+ A_14+ A_19+ A_20+ A_24- A_25+ A_27- A_142- A_159+ 0.1*A_166- A_175- A_181+ 2 *A_217+ A_219+ A_223+ A_224+ A_225+ A_234+ A_259+ A_271;
        varDot(index,132) = A_174- A_175- 2*A_176- 2*A_177- A_179+ A_181- A_184- A_185+ A_186- A_189- A_190- A_199- A_200- A_201  - A_216- A_271+ 0.15*A_274;
        varDot(index,133) = A_19+ 2*A_21- A_23- A_24+ A_25- A_28- A_31- A_32- A_44- A_45+ A_47+ A_50+ A_51+ A_52+ A_55- A_60- A_73 - A_82- A_98- A_102- A_106- A_115- A_120- A_127- A_136- A_156- A_184+ A_223- A_224+ A_226+ A_228;
        varDot(index,134) = A_141- A_142- 2*A_143- 2*A_144- 2*A_145- 2*A_146+ 2*A_147+ A_150- A_152- A_153+ A_155- A_156- A_157  + A_158+ A_159- A_164+ A_165+ 0.46*A_166+ A_172+ A_173- A_199- A_200- A_201+ A_259+ A_264;
        varDot(index,135) = A_23- A_25- A_26- A_27+ 2*A_28- A_29+ A_30+ A_32- A_33- A_34+ A_35+ A_36+ A_38+ A_39- A_46- A_47- A_52 + A_60+ A_61+ A_73+ A_74+ A_82- A_83+ A_84+ A_90+ A_91+ A_92+ 0.96*A_98+ 0.98*A_102+ A_106+ A_111+ 0.84 *A_115+ A_120- A_121+ 0.985*A_127+ A_129+ A_130+ A_131+ 0.956*A_136+ A_156- A_157+ A_158+ A_184- A_185 + A_186- A_223+ A_225+ A_227+ A_229+ 0.667*A_230+ A_239+ A_240+ A_246+ A_253+ A_254+ A_256+ A_262+ A_264 + A_273+ 0.15*A_274;
        varDot(index,136) = A_26- A_28- A_29+ A_30- A_35+ A_37- A_61- A_66- A_74- A_78- A_84- A_96- A_134+ A_159+ A_160+ A_183- A_209 - A_225- A_226+ A_227+ 0.333*A_230+ A_263+ 0.85*A_274;
        varDot(index,137) = A_4+ A_8- A_10- A_11- A_12- A_13- A_14- A_15- 2*A_16+ A_18- A_32- A_34- A_35+ A_38- A_42- A_43+ A_44 + A_55+ A_58- A_59+ A_60+ A_61+ 2*A_62+ A_65+ A_66+ A_68+ 0.13*A_70- A_72+ A_73+ A_74+ A_75- A_80- A_81 + A_85+ 0.82*A_87+ 0.26*A_94- A_97+ 0.96*A_98+ 0.8*A_99- A_101+ 0.98*A_102- A_105+ 0.3*A_107+ A_109+ 1.23 *A_113- A_114+ 0.56*A_115+ 0.32*A_117- A_119+ 0.75*A_120+ 0.875*A_122+ A_123- A_126+ 0.25*A_132- A_135 + 0.956*A_136+ A_137+ A_138- A_149- A_150+ A_151+ 0.94*A_152- A_153+ A_162+ A_164- A_178- A_179+ A_187 + A_190+ A_206+ A_208+ 0.4*A_210- A_213+ 0.667*A_230+ A_231+ A_233+ A_236+ A_237+ A_241+ A_243+ A_244 + A_246+ 0.67*A_247+ A_248+ 0.75*A_249+ 0.67*A_253+ A_255+ A_256;
        varDot(index,138) = A_148+ A_149+ A_151+ 0.06*A_152- A_154+ A_161+ A_162+ A_163+ A_170+ A_171+ A_214- A_260- A_287- A_288 - A_290- A_294- A_296;
    }
}

__device__ void ros_FunTimeDerivative(const double T, double roundoff, double * __restrict__ var, const double * __restrict__ fix, 
                                      const double * __restrict__ rconst, double *dFdT, double *Fcn0, int &Nfun, 
                                      const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
                                      const double * __restrict__ jx,
                                      const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    const double DELTAMIN = 1.0E-6;
    double delta,one_over_delta;

    delta = sqrt(roundoff)*fmax(DELTAMIN,fabs(T));
    one_over_delta = 1.0/delta;

    Fun(var, fix, rconst, dFdT, Nfun, VL_GLO);

    for (int i=0; i < NVAR; i++){
        dFdT(index,i) = (dFdT(index,i) - Fcn0(index,i)) * one_over_delta;
    }
}

__device__  static  int ros_Integrator(double * __restrict__ var, const double * __restrict__ fix, const double Tstart, const double Tend, double &T,
        //  Rosenbrock method coefficients
        const int ros_S, const double * __restrict__ ros_M, const double * __restrict__ ros_E, const double * __restrict__ ros_A, const double * __restrict__  ros_C, 
        const double * __restrict__ ros_Alpha, const double * __restrict__ ros_Gamma, const double ros_ELO, const int * ros_NewF, 
        //  Integration parameters
        const int autonomous, const int vectorTol, const int Max_no_steps, 
        const double roundoff, const double Hmin, const double Hmax, const double Hstart, double &Hexit, 
        const double FacMin, const double FacMax, const double FacRej, const double FacSafe, 
        //  Status parameters
        int &Nfun, int &Njac, int &Nstp, int &Nacc, int &Nrej, int &Ndec, int &Nsol, int &Nsng,
        //  cuda global mem buffers              
        const double * __restrict__ rconst,  const double * __restrict__ absTol, const double * __restrict__ relTol, double * __restrict__ varNew, double * __restrict__ Fcn0, 
        double * __restrict__ K, double * __restrict__ dFdT, double * __restrict__ jac0, double * __restrict__ Ghimj, double * __restrict__ varErr,
        // for update_rconst
        const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
        const double * __restrict__ jx,
        // VL_GLO
        const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    double H, Hnew, HC, HG, Fac; // Tau - not used
    double Err; //*varErr;
    int direction;
    int rejectLastH, rejectMoreH;
    const double DELTAMIN = 1.0E-5;

    //   ~~~>  Initial preparations
    T = Tstart;
    Hexit = 0.0;
    H = fmin(Hstart,Hmax);
    if (fabs(H) <= 10.0*roundoff) 
        H = DELTAMIN;

    if (Tend  >=  Tstart)
    {
        direction = + 1;
    }
    else
    {
        direction = - 1;
    }

    rejectLastH=0;
    rejectMoreH=0;



    //   ~~~> Time loop begins below

    // TimeLoop: 
    while((direction > 0) && ((T- Tend)+ roundoff <= ZERO) || (direction < 0) && ((Tend-T)+ roundoff <= ZERO))
    {
        if (Nstp > Max_no_steps) //  Too many steps
            return -6;
        //  Step size too small
        if (H <= roundoff){  //  Step size too small
            //if (((T+ 0.1*H) == T) || (H <= roundoff)) {
            return -7;
        }

        //   ~~~>  Limit H if necessary to avoid going beyond Tend
        Hexit = H;
        H = fmin(H,fabs(Tend-T));

        //   ~~~>   Compute the function at current time
        Fun(var, fix, rconst, Fcn0, Nfun, VL_GLO);	/// VAR READ - Fcn0 Write

        //   ~~~>  Compute the function derivative with respect to T
        if (!autonomous)
            ros_FunTimeDerivative(T, roundoff, var, fix, rconst, dFdT, Fcn0, Nfun, khet_st, khet_tr, jx,  VL_GLO); /// VAR READ - fcn0 read

        //   ~~~>   Compute the Jacobian at current time
        Jac_sp(var, fix, rconst, jac0, Njac, VL_GLO);   /// VAR READ 

        //   ~~~>  Repeat step calculation until current step accepted
        // UntilAccepted: 
        while(1)
        {
            ros_PrepareMatrix(H, direction, ros_Gamma[0], jac0, Ghimj, Nsng, Ndec, VL_GLO);
            //   ~~~>   Compute the stages
            // Stage: 
            for (int istage=0; istage < ros_S; istage++)
            {
                //   For the 1st istage the function has been computed previously
                if (istage == 0)
                {
                    for (int i=0; i<NVAR; i++){
                        varNew(index,i) = Fcn0(index,i);				// FCN0 Read
                    }
                }
                else if(ros_NewF[istage])
                {
                        for (int i=0; i<NVAR; i++){		
                            varNew(index,i) = var(index,i);
                        }

                    for (int j=0; j < (istage); j++){
                        for (int i=0; i<NVAR; i++){		
                            varNew(index,i) = K(index,j,i)*ros_A[(istage)*(istage-1)/2 + j]  + varNew(index,i);
                        }
                    }
                    Fun(varNew, fix, rconst, varNew, Nfun,VL_GLO); // FCN <- varNew / not overlap 
		} 

		for (int i=0; i<NVAR; i++)		
			K(index,istage,i)  = varNew(index,i);

		for (int j=0; j<(istage); j++)
		{
			HC = ros_C[(istage)*(istage-1)/2 + j]/(direction*H);
			for (int i=0; i<NVAR; i++){
				double tmp = K(index,j,i);
				K(index,istage,i) += tmp*HC;
			}
		}

                if ((!autonomous) && (ros_Gamma[istage] ))
                {
                    HG = direction*H*ros_Gamma[istage];
                    for (int i=0; i<NVAR; i++){
                        K(index,istage,i) += dFdT(index,i)*HG;
		     }
                }
		//	   R   ,RW, RW,  R,        R 
                ros_Solve(Ghimj, K, Nsol, istage, ros_S);


            } // Stage

            //  ~~~>  Compute the new solution
	    for (int i=0; i<NVAR; i++){
		    double tmpNew  = var(index,i); 					/// VAR READ
		    double tmpErr  = ZERO;

		    for (int j=0; j<ros_S; j++){
		    	    double tmp = K(index,j,i);

#ifdef DEBUG
			    if (isnan(tmp)){
			    	printf("Solver detected NAN!");
			    	tmp = 0;
			    }
#endif
			    tmpNew += tmp*ros_M[j];
			    tmpErr += tmp*ros_E[j];
		    }
		    varNew(index,i) = tmpNew;			// varNew is killed
		    varErr(index,i) = tmpErr;
	    }

            Err = ros_ErrorNorm(var, varNew, varErr, absTol, relTol, vectorTol);   /// VAR-varNew READ


//  ~~~> New step size is bounded by FacMin <= Hnew/H <= FacMax
            Fac  = fmin(FacMax,fmax(FacMin,FacSafe/pow(Err,ONE/ros_ELO)));
            Hnew = H*Fac;

//  ~~~>  Check the error magnitude and adjust step size
            Nstp = Nstp+ 1;
            if((Err <= ONE) || (H <= Hmin)) // ~~~> Accept step
            {
                Nacc = Nacc + 1;
                for (int j=0; j<NVAR ; j++)
                    var(index,j) =  fmax(varNew(index,j),ZERO);  /////////// VAR WRITE - last VarNew read

                T = T +  direction*H;
                Hnew = fmax(Hmin,fmin(Hnew,Hmax));
                if (rejectLastH)   // No step size increase after a rejected step
                    Hnew = fmin(Hnew,H);
                rejectLastH = 0;
                rejectMoreH = 0;
                H = Hnew;

            	break;  //  EXIT THE LOOP: WHILE STEP NOT ACCEPTED
            }
            else      // ~~~> Reject step
            {
                if (rejectMoreH)
                    Hnew = H*FacRej;
                rejectMoreH = rejectLastH;
                rejectLastH = 1;
                H = Hnew;
                if (Nacc >= 1)
                    Nrej += 1;
            } //  Err <= 1
        } // UntilAccepted
    } // TimeLoop
//  ~~~> Succesful exit
    return 0; //  ~~~> The integration was successful
}

typedef struct {
 double ros_A[15];
 double ros_C[15];
 int   ros_NewF[8];
 double ros_M[6];
 double ros_E[6];
 double ros_Alpha[6];
 double ros_Gamma[6];
 double ros_ELO;
 int    ros_S;
} ros_t;

/*
 * Lookup tables for different ROS for branch elimination. It is much faster in GPU.
 */
__device__ __constant__  ros_t ros[5] = {
    {       
        {.58578643762690495119831127579030,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, /* ros_A */
        {-1.17157287525380990239662255158060,0,0,0,0,0,0,0,0,0,0,0,0,0,0}, /* ros_C */
        {1,1,0,0,0,0,0,0}, /* ros_NewF */
        {.87867965644035742679746691368545,.29289321881345247559915563789515,0,0,0,0}, /* ros_M */
        {.29289321881345247559915563789515,.29289321881345247559915563789515,0,0,0,0}, /* ros_E */
        {0,1.0,0,0,0,0}, /* ros_Alpha */
        {1.70710678118654752440084436210485,-1.70710678118654752440084436210485,0,0,0,0},  /* ros_Gamma */
        2.0, /* ros_ELO */
        2, /* ros_S*/
    }, /* Ros2 */
    {       
        {1.0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0}, /* ros_A */
        {-0.10156171083877702091975600115545E+01, 0.40759956452537699824805835358067E+01,0.92076794298330791242156818474003E+01,0,0,0,0,0,0,0,0,0,0,0,0}, /* ros_C */
        {1,1,0,0,0,0,0,0}, /* ros_NewF */
        {0.1E+01,0.61697947043828245592553615689730E+01,-0.42772256543218573326238373806514E+00,0,0,0}, /* ros_M */
        {0.5E+00,- 0.29079558716805469821718236208017E+01,0.22354069897811569627360909276199E+00,0,0,0}, /* ros_E */
        {0.0E+00,0.43586652150845899941601945119356E+00,0.43586652150845899941601945119356E+00,0,0,0}, /* ros_Alpha */
        {0.43586652150845899941601945119356E+00,0.24291996454816804366592249683314E+00,0.21851380027664058511513169485832E+01,0,0,0},  /* ros_Gamma */
        3.0, /* ros_ELO */
        3
    }, /* Ros3 */
    {       
        {0.2000000000000000E+01, 0.1867943637803922E+01, 0.2344449711399156E+00, 0.1867943637803922E+01, 0.2344449711399156E+00,0,0,0,0,0,0,0,0,0,0}, /* ros_A */
        {-0.7137615036412310E+01,0.2580708087951457E+01,0.6515950076447975E+00, - 0.2137148994382534E+01, - 0.3214669691237626E+00, - 0.6949742501781779E+00 ,0,0,0,0,0,0,0,0,0}, /* ros_C */
        {1,1,1,0,0,0,0,0}, /* ros_NewF */
        {0.2255570073418735E+01, 0.2870493262186792E+00, 0.4353179431840180E+00, 0.1093502252409163E+01,0,0}, /* ros_M */
        { -0.2815431932141155E+00, -0.7276199124938920E-01, -0.1082196201495311E+00, -0.1093502252409163E+01, 0, 0}, /* ros_E */
        {0.0, 0.1145640000000000E+01, 0.6552168638155900E+00, 0.6552168638155900E+00,0,0}, /* ros_Alpha */
        { 0.5728200000000000E+00, -0.1769193891319233E+01, 0.7592633437920482E+00, -0.1049021087100450E+00,0,0},  /* ros_Gamma */
        4.0, /* ros_ELO */
        4
    }, /* Ros4 */
    {       
        { 0.0E+00, 2.0E+00, 0.0E+00, 2.0E+00, 0.0E+00, 1.0E+00, 0,0,0,0,0,0,0,0,0}, /* ros_A */
        { 4.0E+00, 1.0E+00, - 1.0E+00,  1.0E+00, - 1.0E+00, - 2.66666666666666666666666666666666, 0,0,0,0,0,0,0,0,0}, /* ros_C */
        {1,0,1,1,0,0,0,0}, /* ros_NewF */
        {2.0,0,1.0,1.0,0,0}, /* ros_M */
        {0,0,0,1.0,0,0}, /* ros_E */
        {0,0,1.0,1.0,0,0}, /* ros_Alpha */
        {0.5,1.5,0,0,0,0},  /* ros_Gamma */
        3.0, /* ros_ELO */
        4
    }, /* Rodas3 */

    { 
        {
            0.1544000000000000E+01,  0.9466785280815826E+00, 0.2557011698983284E+00, 0.3314825187068521E+01,
            0.2896124015972201E+01,  0.9986419139977817E+00, 0.1221224509226641E+01, 0.6019134481288629E+01,
            0.1253708332932087E+02, -0.6878860361058950E+00, 0.1221224509226641E+01, 0.6019134481288629E+01,
            0.1253708332932087E+02, -0.6878860361058950E+00, 1.0E+00},  /* ros_A */ 

        {
            -0.5668800000000000E+01, -0.2430093356833875E+01, -0.2063599157091915E+00, -0.1073529058151375E+00,  
            -0.9594562251023355E+01, -0.2047028614809616E+02,  0.7496443313967647E+01, -0.1024680431464352E+02,  
            -0.3399990352819905E+02,  0.1170890893206160E+02,  0.8083246795921522E+01, -0.7981132988064893E+01,  
            -0.3152159432874371E+02,  0.1631930543123136E+02, -0.6058818238834054E+01}, /* ros_C */
        {1,1,1,1,1,1,0,0}, /* ros_NewF */
        {0.1221224509226641E+01,0.6019134481288629E+01,0.1253708332932087E+02,- 0.6878860361058950E+00,1,1}, /* ros_M */
        {0,0,0,0,0,1.0}, /* ros_E */
        {0.000,  0.386,  0.210,  0.630,  1.000, 1.000}, /* ros_Alpha */
        {0.2500000000000000E+00,  -0.1043000000000000E+00,  0.1035000000000000E+00,  0.3620000000000023E-01, 0, 0},  /* ros_Gamma */
        4.0, /* ros_ELO */
        6
    } /* Rodas4 */



};



//__device__ double rconst_local[MAX_VL_GLO*NREACT];

/* Initialize rconst local  */
//__device__ double * rconst_local;


__device__ double k_3rd(double temp, double cair, double k0_300K, double n, double kinf_300K, double m, double fc)
    /*
 *    
 * temp        temperature [K]
 * cair        air concentration [molecules/cm3]
 * k0_300K     low pressure limit at 300 K
 * n           exponent for low pressure limit
 * kinf_300K   high pressure limit at 300 K
 * m           exponent for high pressure limit
 * fc          broadening factor (usually fc=0.6)
 * 
 */
{

    double zt_help, k0_T, kinf_T, k_ratio, k_3rd_r;

    zt_help = 300.0/temp;
    k0_T    = k0_300K   *pow(zt_help,n) *cair;
    kinf_T  = kinf_300K *pow(zt_help,m);
    k_ratio = k0_T/kinf_T;
    k_3rd_r   = k0_T/(1.0+ k_ratio)*pow(fc,1.0/(1.0+ pow(log10(k_ratio),2)));
    return k_3rd_r;
}

__device__ double k_3rd_iupac(double temp, double cair, double k0_300K, double n, double kinf_300K, double m, double fc)
/*
 *    
 * temp        temperature [K]
 * cair        air concentration [molecules/cm3]
 * k0_300K     low pressure limit at 300 K
 * n           exponent for low pressure limit
 * kinf_300K   high pressure limit at 300 K
 * m           exponent for high pressure limit
 * fc          broadening factor (e.g. 0.45 or 0.6...)
 * nu          N
 * 
 */
{   
 
    double zt_help, k0_T, kinf_T, k_ratio, nu, k_3rd_iupac_r;
    zt_help = 300.0/temp;
    k0_T    = k0_300K   *pow(zt_help,n) *cair;
    kinf_T  = kinf_300K *pow(zt_help,m);
    k_ratio = k0_T/kinf_T;
    nu      = 0.75- 1.27*log10(fc);
    k_3rd_iupac_r = k0_T/(1.0+ k_ratio)*pow(fc,1.0/(1.0+ pow(log10(k_ratio)/nu,2)));
    return k_3rd_iupac_r;
}




double * temp_gpu;
double * press_gpu;
double * cair_gpu;


__device__ void  update_rconst(const double * __restrict__ var, 
 			       const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
 			       const double * __restrict__ jx, double * __restrict__ rconst, 
			       const double * __restrict__ temp_gpu, 
			       const double * __restrict__ press_gpu, 
			       const double * __restrict__ cair_gpu, 
			       const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    /* Set local buffer */

    {
        const double temp_loc  = temp_gpu[index];
        const double press_loc = press_gpu[index];
        const double cair_loc  = cair_gpu[index];

        double k_HO2_HO2, k_NO3_NO2, k_NO2_HO2, k_HNO3_OH, k_CH3OOH_OH, k_ClO_ClO, k_BrO_NO2, k_I_NO2, k_DMS_OH, k_CH2OO_SO2, k_O3s, beta_null_CH3NO3, beta_inf_CH3NO3, beta_CH3NO3, k_NO2_CH3O2, k_C6H5O2_NO2, k_CH2OO_NO2, beta_C2H5NO3, alpha_NO_HO2, beta_NO_HO2, k0_NO_HO2, k2d_NO_HO2, k1d_NO_HO2, k2w_NO_HO2, k1w_NO_HO2, k_PrO2_HO2, k_PrO2_NO, k_PrO2_CH3O2, G7402a_yield, k_CH3CO3_NO2, k_PAN_M, KRO2NO, KRO2HO2[12], KAPNO, KRO2NO3, KNO3AL, KAPHO2, k_CH3O2, k_RO2RCO3, k_RO2pRO2, k_RO2sRO2, k_RO2tRO2, k_RO2pORO2, k_RO2sORO2, k_RO2tORO2, k_RO2LISOPACO2, k_RO2ISOPBO2, k_RO2ISOPDO2, k_p, k_s, k_t, k_rohro, k_co2h, k_adp, k_ads, k_adt, KHSB, KHSD, K16HSZ14, K16HSZ41, K16HS, K15HSDHB, K14HSAL, K15HS24VYNAL, K15HS42VYNAL, KHYDEC, k_CH2CHOH_OH_HCOOH, k_CH2CHOH_OH_ALD, k_CH2CHOH_HCOOH, k_ALD_HCOOH, J_IC3H7NO3, J_ACETOL, J_HPALD, J_KETENE, RO2, k1_RO2RCO3, k1_RO2pRO2, k1_RO2sRO2, k1_RO2tRO2, k1_RO2pORO2, k1_RO2sORO2, k1_RO2tORO2, k1_RO2LISOPACO2, k1_RO2ISOPBO2, k1_RO2ISOPDO2;

        k_HO2_HO2 = (3.0E-13 *exp(460. / temp_loc)+ 2.1E-33 *exp(920. / temp_loc) *cair_loc) * (1.+ 1.4E-21 *exp(2200. / temp_loc) *var(index,ind_H2O));
        k_NO3_NO2 = k_3rd(temp_loc , cair_loc , 2.4E-30 , 3.0 , 1.6E-12 , - 0.1 , 0.6);
        k_NO2_HO2 = k_3rd(temp_loc , cair_loc , 1.9E-31 , 3.4 , 4.0E-12 , 0.3 , 0.6);
        k_HNO3_OH = 1.32E-14 *exp(527. / temp_loc) + 1. / (1. / (7.39E-32 *exp(453. / temp_loc) *cair_loc) + 1. / (9.73E-17 *exp(1910. / temp_loc)));
        k_CH3OOH_OH = 5.3E-12 *exp(190. / temp_loc);
        k_ClO_ClO = k_3rd(temp_loc , cair_loc , 1.9E-32 , 3.6 , 3.7E-12 , 1.6 , 0.6);
        k_BrO_NO2 = k_3rd_iupac(temp_loc , cair_loc , 4.7E-31 , 3.1 , 1.8E-11 , 0.0 , 0.4);
        k_I_NO2 = k_3rd_iupac(temp_loc , cair_loc , 3.0E-31 , 1.0 , 6.6E-11 , 0.0 , 0.63);
        k_DMS_OH = 1.E-9 *exp(5820. / temp_loc) *var(index,ind_O2) / (1.E30+ 5. *exp(6280. / temp_loc) *var(index,ind_O2));
        k_CH2OO_SO2 = 3.66E-11;
        k_O3s = (1.7E-12 *exp(- 940. / temp_loc)) *var(index,ind_OH) + (1.E-14 *exp(- 490. / temp_loc)) *var(index,ind_HO2) + jx(index,ip_O1D) *2.2E-10 *var(index,ind_H2O) / (3.2E-11 *exp(70. / temp_loc) *var(index,ind_O2) + 1.8E-11 *exp(110. / temp_loc) *var(index,ind_N2) + 2.2E-10 *var(index,ind_H2O));
        beta_null_CH3NO3 = 0.00295 + 5.15E-22 *cair_loc * pow(temp_loc / 298, 7.4);
        beta_inf_CH3NO3 = 0.022;
        beta_CH3NO3 = (beta_null_CH3NO3 *beta_inf_CH3NO3) / (beta_null_CH3NO3 + beta_inf_CH3NO3) / 10.;
        k_NO2_CH3O2 = k_3rd(temp_loc , cair_loc , 1.0E-30 , 4.8 , 7.2E-12 , 2.1 , 0.6);
        k_C6H5O2_NO2 = k_NO2_CH3O2;
        k_CH2OO_NO2 = 4.25E-12;
        beta_C2H5NO3 = (1- 1 / (1+ 1.E-2 *(3.88e-3 *cair_loc / 2.46e19 *760.+ .365) *(1+ 1500. *(1 / temp_loc - 1 / 298.))));
        alpha_NO_HO2 = var(index,ind_H2O) *6.6E-27 *temp_loc *exp(3700. / temp_loc);
        beta_NO_HO2 = max(((530. / temp_loc)+ (press_loc *4.8004E-6)- 1.73) *0.01 , 0.);
        k0_NO_HO2 = 3.5E-12 *exp(250. / temp_loc);
        k2d_NO_HO2 = (beta_NO_HO2 *k0_NO_HO2) / (1.+ beta_NO_HO2);
        k1d_NO_HO2 = k0_NO_HO2 - k2d_NO_HO2;
        k2w_NO_HO2 = (beta_NO_HO2 *k0_NO_HO2 *(1.+ 42. *alpha_NO_HO2))/ ((1.+ alpha_NO_HO2) *(1.+ beta_NO_HO2));
        k1w_NO_HO2 = k0_NO_HO2 - k2w_NO_HO2;
        k_PrO2_HO2 = 1.9E-13 *exp(1300. / temp_loc);
        k_PrO2_NO = 2.7E-12 *exp(360. / temp_loc);
        k_PrO2_CH3O2 = 9.46E-14 *exp(431. / temp_loc);
        G7402a_yield = 0.8 / 1.1;
        k_CH3CO3_NO2 = k_3rd(temp_loc , cair_loc , 9.7E-29 , 5.6 , 9.3E-12 , 1.5 , 0.6);
        k_PAN_M = k_CH3CO3_NO2 / (9.0E-29 *exp(14000. / temp_loc));
        KRO2NO = 2.54E-12 *exp(360. / temp_loc);

        /*KRO2HO2(:) = 2.91E-13 *exp(1300. / temp_loc) *(1.- exp(- 0.245 *(nC(:))));*/
        for (int ii=0;ii<12;ii++) {
            KRO2HO2[ii] = 2.91E-13 *exp(1300. / temp_loc) * (1.- exp(- 0.245 *float(ii+1)));
        }

        KAPNO = 8.10E-12 *exp(270. / temp_loc);
        KRO2NO3 = 2.50E-12;
        KNO3AL = 1.4E-12 *exp(- 1900. / temp_loc);
        KAPHO2 = 5.20E-13 *exp(980. / temp_loc) *1.865;
        k_CH3O2 = 1.03E-13 *exp(365. / temp_loc);
        k_RO2RCO3 = 2. *2.E-12 *exp(500. / temp_loc);
        k_RO2pRO2 = 2. * pow(1.E-12 *k_CH3O2, .5);
        k_RO2sRO2 = 2. * pow(1.6E-12 *exp(- 2200. / temp_loc) *k_CH3O2, .5);
        k_RO2tRO2 = 2. *3.8E-13 *exp(- 1430. / temp_loc);
        k_RO2pORO2 = 2. *7.5E-13 *exp(500. / temp_loc);
        k_RO2sORO2 = 2. * pow(7.7E-15 *exp(1330. / temp_loc) *k_CH3O2, .5);
        k_RO2tORO2 = 2. * pow(4.7E-13 *exp(- 1420. / temp_loc) *k_CH3O2, .5);
        k_RO2LISOPACO2 = 2. * pow((2.8E-12+ 3.9E-12) / 2. *k_CH3O2, .5);
        k_RO2ISOPBO2 = 2. * pow(6.9E-14 *k_CH3O2, .5);
        k_RO2ISOPDO2 = 2. * pow(4.8E-12 *k_CH3O2, .5);
        k_p = 4.49E-18 *temp_loc *temp_loc *exp(- 320. / temp_loc);
        k_s = 4.50E-18 *temp_loc *temp_loc *exp(253. / temp_loc);
        k_t = 2.12E-18 *temp_loc *temp_loc *exp(696. / temp_loc);
        k_rohro = 2.1E-18 *temp_loc *temp_loc *exp(- 85. / temp_loc);
        k_co2h = .7 *4.2E-14 *exp(850. / temp_loc);
        k_adp = 4.5E-12 * pow(temp_loc / 300., - 0.85);
        k_ads = .25 *(1.1E-11 *exp(485. / temp_loc)+ 1.0E-11 *exp(553. / temp_loc));
        k_adt = 1.922E-11 *exp(450. / temp_loc) - k_ads;
        KHSB = 1.52E11 *exp(- 9512. / temp_loc) *1.;
        KHSD = 6.08E10 *exp(- 8893. / temp_loc) *1.;
        K16HSZ14 = 2.28E9 *exp(- 6764 / temp_loc) *0.28;
        K16HSZ41 = 1.23E9 *exp(- 6186 / temp_loc) *0.28;
        K16HS = pow(K16HSZ14 *K16HSZ41, .5);
        K15HSDHB = 5.;
        K14HSAL = 2.9E7 *exp(- 1 *(5297+ 705) / temp_loc);
        K15HS24VYNAL = K16HSZ14 *exp(- 3500 / (1.987 *temp_loc));
        K15HS42VYNAL = K16HSZ41 *exp(- 3500 / (1.987 *temp_loc));
        KHYDEC = 6.e14 *exp(- 16000. / (1.98588 *temp_loc));
        k_CH2CHOH_OH_HCOOH = 4.3E-11;
        k_CH2CHOH_OH_ALD = 2.4E-11;
        k_CH2CHOH_HCOOH = 4.67E-26 * pow(temp_loc, 3.286 *exp(4509. / (1.987 *temp_loc)));
        k_ALD_HCOOH = 1.17E-19 * pow(temp_loc, 1.209 *exp(- 556. / (1.987 *temp_loc)));
        J_IC3H7NO3 = 3.7 *jx(index,ip_PAN);
        J_ACETOL = 0.65 *0.11 *jx(index,ip_CHOH);
        J_HPALD = (jx(index,ip_CH3OOH)+ jx(index,ip_MACR) / (2. *1.95E-3));
        J_KETENE = jx(index,ip_MVK) / 0.004;
        RO2 = 0.;
        if (ind_LISOPACO2>0) RO2 = RO2 + var(index,ind_LISOPACO2);
        if (ind_LDISOPACO2>0) RO2 = RO2 + var(index,ind_LDISOPACO2);
        if (ind_ISOPBO2>0) RO2 = RO2 + var(index,ind_ISOPBO2);
        if (ind_ISOPDO2>0) RO2 = RO2 + var(index,ind_ISOPDO2);
        if (ind_LISOPEFO2>0) RO2 = RO2 + var(index,ind_LISOPEFO2);
        if (ind_NISOPO2>0) RO2 = RO2 + var(index,ind_NISOPO2);
        if (ind_LHC4ACCO3>0) RO2 = RO2 + var(index,ind_LHC4ACCO3);
        if (ind_LC578O2>0) RO2 = RO2 + var(index,ind_LC578O2);
        if (ind_C59O2>0) RO2 = RO2 + var(index,ind_C59O2);
        if (ind_LNISO3>0) RO2 = RO2 + var(index,ind_LNISO3);
        if (ind_CH3O2>0) RO2 = RO2 + var(index,ind_CH3O2);
        if (ind_HOCH2O2>0) RO2 = RO2 + var(index,ind_HOCH2O2);
        if (ind_CH3CO3>0) RO2 = RO2 + var(index,ind_CH3CO3);
        if (ind_C2H5O2>0) RO2 = RO2 + var(index,ind_C2H5O2);
        if (ind_HOCH2CO3>0) RO2 = RO2 + var(index,ind_HOCH2CO3);
        if (ind_HYPROPO2>0) RO2 = RO2 + var(index,ind_HYPROPO2);
        if (ind_LBUT1ENO2>0) RO2 = RO2 + var(index,ind_LBUT1ENO2);
        if (ind_BUT2OLO2>0) RO2 = RO2 + var(index,ind_BUT2OLO2);
        if (ind_HCOCO3>0) RO2 = RO2 + var(index,ind_HCOCO3);
        if (ind_CO2H3CO3>0) RO2 = RO2 + var(index,ind_CO2H3CO3);
        if (ind_LHMVKABO2>0) RO2 = RO2 + var(index,ind_LHMVKABO2);
        if (ind_MACO3>0) RO2 = RO2 + var(index,ind_MACO3);
        if (ind_MACRO2>0) RO2 = RO2 + var(index,ind_MACRO2);
        if (ind_PRONO3BO2>0) RO2 = RO2 + var(index,ind_PRONO3BO2);
        if (ind_HOCH2CH2O2>0) RO2 = RO2 + var(index,ind_HOCH2CH2O2);
        if (ind_CH3COCH2O2>0) RO2 = RO2 + var(index,ind_CH3COCH2O2);
        if (ind_IC3H7O2>0) RO2 = RO2 + var(index,ind_IC3H7O2);
        if (ind_NC3H7O2>0) RO2 = RO2 + var(index,ind_NC3H7O2);
        if (ind_LC4H9O2>0) RO2 = RO2 + var(index,ind_LC4H9O2);
        if (ind_TC4H9O2>0) RO2 = RO2 + var(index,ind_TC4H9O2);
        if (ind_LMEKO2>0) RO2 = RO2 + var(index,ind_LMEKO2);
        if (ind_HCOCH2O2>0) RO2 = RO2 + var(index,ind_HCOCH2O2);
        if (ind_EZCH3CO2CHCHO>0) RO2 = RO2 + var(index,ind_EZCH3CO2CHCHO);
        if (ind_EZCHOCCH3CHO2>0) RO2 = RO2 + var(index,ind_EZCHOCCH3CHO2);
        if (ind_CH3COCHO2CHO>0) RO2 = RO2 + var(index,ind_CH3COCHO2CHO);
        if (ind_HCOCO2CH3CHO>0) RO2 = RO2 + var(index,ind_HCOCO2CH3CHO);
        if (ind_C1ODC3O2C4OOH>0) RO2 = RO2 + var(index,ind_C1ODC3O2C4OOH);
        if (ind_C1OOHC2O2C4OD>0) RO2 = RO2 + var(index,ind_C1OOHC2O2C4OD);
        if (ind_C1ODC2O2C4OD>0) RO2 = RO2 + var(index,ind_C1ODC2O2C4OD);
        if (ind_ISOPBDNO3O2>0) RO2 = RO2 + var(index,ind_ISOPBDNO3O2);
        if (ind_LISOPACNO3O2>0) RO2 = RO2 + var(index,ind_LISOPACNO3O2);
        if (ind_DB1O2>0) RO2 = RO2 + var(index,ind_DB1O2);
        if (ind_DB2O2>0) RO2 = RO2 + var(index,ind_DB2O2);
        if (ind_LME3FURANO2>0) RO2 = RO2 + var(index,ind_LME3FURANO2);
        if (ind_NO3CH2CO3>0) RO2 = RO2 + var(index,ind_NO3CH2CO3);
        if (ind_CH3COCO3>0) RO2 = RO2 + var(index,ind_CH3COCO3);
        if (ind_ZCO3C23DBCOD>0) RO2 = RO2 + var(index,ind_ZCO3C23DBCOD);
        if (ind_IBUTOLBO2>0) RO2 = RO2 + var(index,ind_IBUTOLBO2);
        if (ind_IPRCO3>0) RO2 = RO2 + var(index,ind_IPRCO3);
        if (ind_IC4H9O2>0) RO2 = RO2 + var(index,ind_IC4H9O2);
        if (ind_LMBOABO2>0) RO2 = RO2 + var(index,ind_LMBOABO2);
        if (ind_IPRHOCO3>0) RO2 = RO2 + var(index,ind_IPRHOCO3);
        if (ind_LNMBOABO2>0) RO2 = RO2 + var(index,ind_LNMBOABO2);
        if (ind_NC4OHCO3>0) RO2 = RO2 + var(index,ind_NC4OHCO3);
        if (ind_LAPINABO2>0) RO2 = RO2 + var(index,ind_LAPINABO2);
        if (ind_C96O2>0) RO2 = RO2 + var(index,ind_C96O2);
        if (ind_C97O2>0) RO2 = RO2 + var(index,ind_C97O2);
        if (ind_C98O2>0) RO2 = RO2 + var(index,ind_C98O2);
        if (ind_C85O2>0) RO2 = RO2 + var(index,ind_C85O2);
        if (ind_C86O2>0) RO2 = RO2 + var(index,ind_C86O2);
        if (ind_PINALO2>0) RO2 = RO2 + var(index,ind_PINALO2);
        if (ind_C96CO3>0) RO2 = RO2 + var(index,ind_C96CO3);
        if (ind_C89CO3>0) RO2 = RO2 + var(index,ind_C89CO3);
        if (ind_C85CO3>0) RO2 = RO2 + var(index,ind_C85CO3);
        if (ind_OHMENTHEN6ONEO2>0) RO2 = RO2 + var(index,ind_OHMENTHEN6ONEO2);
        if (ind_C511O2>0) RO2 = RO2 + var(index,ind_C511O2);
        if (ind_C106O2>0) RO2 = RO2 + var(index,ind_C106O2);
        if (ind_CO235C6CO3>0) RO2 = RO2 + var(index,ind_CO235C6CO3);
        if (ind_CHOC3COCO3>0) RO2 = RO2 + var(index,ind_CHOC3COCO3);
        if (ind_CO235C6O2>0) RO2 = RO2 + var(index,ind_CO235C6O2);
        if (ind_C716O2>0) RO2 = RO2 + var(index,ind_C716O2);
        if (ind_C614O2>0) RO2 = RO2 + var(index,ind_C614O2);
        if (ind_HCOCH2CO3>0) RO2 = RO2 + var(index,ind_HCOCH2CO3);
        if (ind_BIACETO2>0) RO2 = RO2 + var(index,ind_BIACETO2);
        if (ind_CO23C4CO3>0) RO2 = RO2 + var(index,ind_CO23C4CO3);
        if (ind_C109O2>0) RO2 = RO2 + var(index,ind_C109O2);
        if (ind_C811CO3>0) RO2 = RO2 + var(index,ind_C811CO3);
        if (ind_C89O2>0) RO2 = RO2 + var(index,ind_C89O2);
        if (ind_C812O2>0) RO2 = RO2 + var(index,ind_C812O2);
        if (ind_C813O2>0) RO2 = RO2 + var(index,ind_C813O2);
        if (ind_C721CO3>0) RO2 = RO2 + var(index,ind_C721CO3);
        if (ind_C721O2>0) RO2 = RO2 + var(index,ind_C721O2);
        if (ind_C722O2>0) RO2 = RO2 + var(index,ind_C722O2);
        if (ind_C44O2>0) RO2 = RO2 + var(index,ind_C44O2);
        if (ind_C512O2>0) RO2 = RO2 + var(index,ind_C512O2);
        if (ind_C513O2>0) RO2 = RO2 + var(index,ind_C513O2);
        if (ind_CHOC3COO2>0) RO2 = RO2 + var(index,ind_CHOC3COO2);
        if (ind_C312COCO3>0) RO2 = RO2 + var(index,ind_C312COCO3);
        if (ind_HOC2H4CO3>0) RO2 = RO2 + var(index,ind_HOC2H4CO3);
        if (ind_LNAPINABO2>0) RO2 = RO2 + var(index,ind_LNAPINABO2);
        if (ind_C810O2>0) RO2 = RO2 + var(index,ind_C810O2);
        if (ind_C514O2>0) RO2 = RO2 + var(index,ind_C514O2);
        if (ind_CHOCOCH2O2>0) RO2 = RO2 + var(index,ind_CHOCOCH2O2);
        if (ind_ROO6R1O2>0) RO2 = RO2 + var(index,ind_ROO6R1O2);
        if (ind_ROO6R3O2>0) RO2 = RO2 + var(index,ind_ROO6R3O2);
        if (ind_RO6R1O2>0) RO2 = RO2 + var(index,ind_RO6R1O2);
        if (ind_RO6R3O2>0) RO2 = RO2 + var(index,ind_RO6R3O2);
        if (ind_BPINAO2>0) RO2 = RO2 + var(index,ind_BPINAO2);
        if (ind_C8BCO2>0) RO2 = RO2 + var(index,ind_C8BCO2);
        if (ind_NOPINDO2>0) RO2 = RO2 + var(index,ind_NOPINDO2);
        if (ind_LNBPINABO2>0) RO2 = RO2 + var(index,ind_LNBPINABO2);
        if (ind_BZBIPERO2>0) RO2 = RO2 + var(index,ind_BZBIPERO2);
        if (ind_C6H5CH2O2>0) RO2 = RO2 + var(index,ind_C6H5CH2O2);
        if (ind_TLBIPERO2>0) RO2 = RO2 + var(index,ind_TLBIPERO2);
        if (ind_BZEMUCCO3>0) RO2 = RO2 + var(index,ind_BZEMUCCO3);
        if (ind_BZEMUCO2>0) RO2 = RO2 + var(index,ind_BZEMUCO2);
        if (ind_C5DIALO2>0) RO2 = RO2 + var(index,ind_C5DIALO2);
        if (ind_NPHENO2>0) RO2 = RO2 + var(index,ind_NPHENO2);
        if (ind_PHENO2>0) RO2 = RO2 + var(index,ind_PHENO2);
        if (ind_CRESO2>0) RO2 = RO2 + var(index,ind_CRESO2);
        if (ind_NCRESO2>0) RO2 = RO2 + var(index,ind_NCRESO2);
        if (ind_TLEMUCCO3>0) RO2 = RO2 + var(index,ind_TLEMUCCO3);
        if (ind_TLEMUCO2>0) RO2 = RO2 + var(index,ind_TLEMUCO2);
        if (ind_C615CO2O2>0) RO2 = RO2 + var(index,ind_C615CO2O2);
        if (ind_MALDIALCO3>0) RO2 = RO2 + var(index,ind_MALDIALCO3);
        if (ind_EPXDLCO3>0) RO2 = RO2 + var(index,ind_EPXDLCO3);
        if (ind_C3DIALO2>0) RO2 = RO2 + var(index,ind_C3DIALO2);
        if (ind_MALDIALO2>0) RO2 = RO2 + var(index,ind_MALDIALO2);
        if (ind_C6H5O2>0) RO2 = RO2 + var(index,ind_C6H5O2);
        if (ind_C6H5CO3>0) RO2 = RO2 + var(index,ind_C6H5CO3);
        if (ind_OXYL1O2>0) RO2 = RO2 + var(index,ind_OXYL1O2);
        if (ind_C5CO14O2>0) RO2 = RO2 + var(index,ind_C5CO14O2);
        if (ind_NBZFUO2>0) RO2 = RO2 + var(index,ind_NBZFUO2);
        if (ind_BZFUO2>0) RO2 = RO2 + var(index,ind_BZFUO2);
        if (ind_HCOCOHCO3>0) RO2 = RO2 + var(index,ind_HCOCOHCO3);
        if (ind_CATEC1O2>0) RO2 = RO2 + var(index,ind_CATEC1O2);
        if (ind_MCATEC1O2>0) RO2 = RO2 + var(index,ind_MCATEC1O2);
        if (ind_C5DICARBO2>0) RO2 = RO2 + var(index,ind_C5DICARBO2);
        if (ind_NTLFUO2>0) RO2 = RO2 + var(index,ind_NTLFUO2);
        if (ind_TLFUO2>0) RO2 = RO2 + var(index,ind_TLFUO2);
        if (ind_NPHEN1O2>0) RO2 = RO2 + var(index,ind_NPHEN1O2);
        if (ind_NNCATECO2>0) RO2 = RO2 + var(index,ind_NNCATECO2);
        if (ind_NCATECO2>0) RO2 = RO2 + var(index,ind_NCATECO2);
        if (ind_NBZQO2>0) RO2 = RO2 + var(index,ind_NBZQO2);
        if (ind_PBZQO2>0) RO2 = RO2 + var(index,ind_PBZQO2);
        if (ind_NPTLQO2>0) RO2 = RO2 + var(index,ind_NPTLQO2);
        if (ind_PTLQO2>0) RO2 = RO2 + var(index,ind_PTLQO2);
        if (ind_NCRES1O2>0) RO2 = RO2 + var(index,ind_NCRES1O2);
        if (ind_MNNCATECO2>0) RO2 = RO2 + var(index,ind_MNNCATECO2);
        if (ind_MNCATECO2>0) RO2 = RO2 + var(index,ind_MNCATECO2);
        if (ind_MECOACETO2>0) RO2 = RO2 + var(index,ind_MECOACETO2);
        if (ind_CO2H3CO3>0) RO2 = RO2 + var(index,ind_CO2H3CO3);
        if (ind_MALANHYO2>0) RO2 = RO2 + var(index,ind_MALANHYO2);
        if (ind_NDNPHENO2>0) RO2 = RO2 + var(index,ind_NDNPHENO2);
        if (ind_DNPHENO2>0) RO2 = RO2 + var(index,ind_DNPHENO2);
        if (ind_NDNCRESO2>0) RO2 = RO2 + var(index,ind_NDNCRESO2);
        if (ind_DNCRESO2>0) RO2 = RO2 + var(index,ind_DNCRESO2);
        if (ind_C5CO2OHCO3>0) RO2 = RO2 + var(index,ind_C5CO2OHCO3);
        if (ind_C6CO2OHCO3>0) RO2 = RO2 + var(index,ind_C6CO2OHCO3);
        if (ind_MMALANHYO2>0) RO2 = RO2 + var(index,ind_MMALANHYO2);
        if (ind_ACCOMECO3>0) RO2 = RO2 + var(index,ind_ACCOMECO3);
        if (ind_C4CO2DBCO3>0) RO2 = RO2 + var(index,ind_C4CO2DBCO3);
        if (ind_C5CO2DBCO3>0) RO2 = RO2 + var(index,ind_C5CO2DBCO3);
        if (ind_NSTYRENO2>0) RO2 = RO2 + var(index,ind_NSTYRENO2);
        if (ind_STYRENO2>0) RO2 = RO2 + var(index,ind_STYRENO2);
        k1_RO2RCO3 = RO2 *k_RO2RCO3;
        k1_RO2pRO2 = RO2 *k_RO2pRO2;
        k1_RO2sRO2 = RO2 *k_RO2sRO2;
        k1_RO2tRO2 = RO2 *k_RO2tRO2;
        k1_RO2pORO2 = RO2 *k_RO2pORO2;
        k1_RO2sORO2 = RO2 *k_RO2sORO2;
        k1_RO2tORO2 = RO2 *k_RO2tORO2;
        k1_RO2LISOPACO2 = RO2 *k_RO2LISOPACO2;
        k1_RO2ISOPBO2 = RO2 *k_RO2ISOPBO2;
        k1_RO2ISOPDO2 = RO2 *k_RO2ISOPDO2;

        rconst(index,0) = (3.3E-11 *exp(55. / temp_loc));
        rconst(index,1) = (6.0E-34 *( pow(temp_loc / 300., - 2.4) )*cair_loc);
        rconst(index,3) = (8.0E-12 *exp(- 2060. / temp_loc));
        rconst(index,4) = (k_3rd(temp_loc , cair_loc , 4.4E-32 , 1.3 , 7.5E-11 , - 0.2 , 0.6));
        rconst(index,5) = (1.4E-10 *exp(- 470. / temp_loc));
        rconst(index,7) = (1.8E-11 *exp(180. / temp_loc));
        rconst(index,8) = (1.7E-12 *exp(- 940. / temp_loc));
        rconst(index,9) = (2.8E-12 *exp(- 1800. / temp_loc));
        rconst(index,10) = (3.E-11 *exp(200. / temp_loc));
        rconst(index,11) = (1.E-14 *exp(- 490. / temp_loc));
        rconst(index,15) = (4.8E-11 *exp(250. / temp_loc));
        rconst(index,16) = (k_HO2_HO2);
        rconst(index,17) = (1.63E-10 *exp(60. / temp_loc));
        rconst(index,19) = (1.5E-11 *exp(- 3600. / temp_loc));
        rconst(index,20) = (2.15E-11 *exp(110. / temp_loc));
        rconst(index,21) = (7.259E-11 *exp(20. / temp_loc));
        rconst(index,22) = (4.641E-11 *exp(20. / temp_loc));
        rconst(index,23) = (3.0E-12 *exp(- 1500. / temp_loc));
        rconst(index,24) = (2.1E-11 *exp(100. / temp_loc));
        rconst(index,25) = (5.1E-12 *exp(210. / temp_loc));
        rconst(index,26) = (1.2E-13 *exp(- 2450. / temp_loc));
        rconst(index,27) = (5.8E-12 *exp(220. / temp_loc));
        rconst(index,28) = (1.5E-11 *exp(170. / temp_loc));
        rconst(index,29) = (k_NO3_NO2);
        rconst(index,30) = (k_NO3_NO2 / (5.8E-27 *exp(10840. / temp_loc)));
        rconst(index,31) = (k_3rd(temp_loc , cair_loc , 7.0E-31 , 2.6 , 3.6E-11 , 0.1 , 0.6));
        rconst(index,32) = (3.3E-12 *exp(270. / temp_loc));
        rconst(index,33) = (k_3rd(temp_loc , cair_loc , 1.8E-30 , 3.0 , 2.8E-11 , 0. , 0.6));
        rconst(index,34) = (k_NO2_HO2);
        rconst(index,36) = (1.8E-11 *exp(- 390. / temp_loc));
        rconst(index,37) = (k_HNO3_OH);
        rconst(index,38) = (k_NO2_HO2 / (2.1E-27 *exp(10900. / temp_loc)));
        rconst(index,39) = (1.3E-12 *exp(380. / temp_loc));
        rconst(index,40) = (1.7E-12 *exp(- 710. / temp_loc));
        rconst(index,41) = (4.3E-12 *exp(- 930. / temp_loc));
        rconst(index,42) = (4.8E-07 *exp(- 628. / temp_loc) * pow(temp_loc, - 1.32) );
        rconst(index,43) = (9.4E-09 *exp(- 356. / temp_loc) * pow(temp_loc, - 1.12) );
        rconst(index,44) = (1.92E-12 *( pow(temp_loc / 298., - 1.5) ));
        rconst(index,45) = (1.41E-11 *( pow(temp_loc / 298., - 1.5) ));
        rconst(index,46) = (1.2E-11 *( pow(temp_loc / 298., - 2.0) ));
        rconst(index,47) = (0.8E-11 *( pow(temp_loc / 298., - 2.0) ));
        rconst(index,50) = (8.0E-11 *exp(- 500. / temp_loc));
        rconst(index,51) = (1.66E-12 *exp(- 1500. / temp_loc));
        rconst(index,52) = (1.0E-12 *exp(- 1000. / temp_loc));
        rconst(index,54) = (4.13E-11 *exp(- 2138. / temp_loc));
        rconst(index,55) = (3.65E-14 *exp(- 4600. / temp_loc));
        rconst(index,57) = (1.85E-20 *exp(2.82 *log(temp_loc)- 987. / temp_loc));
        rconst(index,58) = (2.9E-12 *exp(- 345. / temp_loc));
        rconst(index,59) = (4.1E-13 *exp(750. / temp_loc));
        rconst(index,60) = (2.8E-12 *exp(300. / temp_loc));
        rconst(index,62) = (9.5E-14 *exp(390. / temp_loc) / (1.+ 1. / 26.2 *exp(1130. / temp_loc)));
        rconst(index,63) = (9.5E-14 *exp(390. / temp_loc) / (1.+ 26.2 *exp(- 1130. / temp_loc)));
        rconst(index,64) = (k_CH3OOH_OH);
        rconst(index,65) = (9.52E-18 *exp(2.03 *log(temp_loc)+ 636. / temp_loc));
        rconst(index,66) = (3.4E-13 *exp(- 1900. / temp_loc));
        rconst(index,67) = ((1.57E-13+ cair_loc *3.54E-33));
        rconst(index,69) = (1.49E-17 *temp_loc *temp_loc *exp(- 499. / temp_loc));
        rconst(index,70) = (1.2E-14 *exp(- 2630. / temp_loc));
        rconst(index,71) = (k_3rd(temp_loc , cair_loc , 1.0E-28 , 4.5 , 7.5E-12 , 0.85 , 0.6));
        rconst(index,72) = (7.5E-13 *exp(700. / temp_loc));
        rconst(index,73) = (2.6E-12 *exp(365. / temp_loc));
        rconst(index,75) = (1.6E-13 *exp(195. / temp_loc));
        rconst(index,76) = (k_CH3OOH_OH);
        rconst(index,77) = (4.4E-12 *exp(365. / temp_loc));
        rconst(index,78) = (1.4E-12 *exp(- 1900. / temp_loc));
        rconst(index,79) = (4.2E-14 *exp(855. / temp_loc));
        rconst(index,80) = (4.3E-13 *exp(1040. / temp_loc) / (1.+ 1. / 37. *exp(660. / temp_loc)));
        rconst(index,81) = (4.3E-13 *exp(1040. / temp_loc) / (1.+ 37. *exp(- 660. / temp_loc)));
        rconst(index,82) = (8.1E-12 *exp(270. / temp_loc));
        rconst(index,83) = (k_CH3CO3_NO2);
        rconst(index,85) = (0.9 *2.0E-12 *exp(500. / temp_loc));
        rconst(index,86) = (0.1 *2.0E-12 *exp(500. / temp_loc));
        rconst(index,87) = (4.9E-12 *exp(211. / temp_loc));
        rconst(index,88) = (2.5E-12 *exp(500. / temp_loc));
        rconst(index,89) = (0.6 *k_CH3OOH_OH);
        rconst(index,90) = (5.6E-12 *exp(270. / temp_loc));
        rconst(index,91) = (9.50E-13 *exp(- 650. / temp_loc));
        rconst(index,92) = (k_PAN_M);
        rconst(index,93) = (1.65E-17 *temp_loc *temp_loc *exp(- 87. / temp_loc));
        rconst(index,94) = (6.5E-15 *exp(- 1900. / temp_loc));
        rconst(index,95) = (k_3rd(temp_loc , cair_loc , 8.E-27 , 3.5 , 3.E-11 , 0. , 0.5));
        rconst(index,96) = (4.6E-13 *exp(- 1155. / temp_loc));
        rconst(index,97) = (k_PrO2_HO2);
        rconst(index,98) = (k_PrO2_NO);
        rconst(index,99) = (k_PrO2_CH3O2);
        rconst(index,100) = (k_CH3OOH_OH);
        rconst(index,101) = (6.5E-13 *exp(650. / temp_loc));
        rconst(index,102) = (4.2E-12 *exp(180. / temp_loc));
        rconst(index,103) = (3.8E-12 *exp(200. / temp_loc));
        rconst(index,104) = (1.33E-13+ 3.82E-11 *exp(- 2000. / temp_loc));
        rconst(index,105) = (8.6E-13 *exp(700. / temp_loc));
        rconst(index,106) = (2.9E-12 *exp(300. / temp_loc));
        rconst(index,107) = (7.5E-13 *exp(500. / temp_loc));
        rconst(index,108) = (k_CH3OOH_OH);
        rconst(index,109) = (2.15E-12 *exp(305. / temp_loc));
        rconst(index,110) = (8.4E-13 *exp(830. / temp_loc));
        rconst(index,111) = (6.2E-13 *exp(- 230. / temp_loc));
        rconst(index,112) = (1.81E-17 *temp_loc *temp_loc *exp(114. / temp_loc));
        rconst(index,113) = (k_PrO2_CH3O2);
        rconst(index,114) = (k_PrO2_HO2);
        rconst(index,115) = (k_PrO2_NO);
        rconst(index,116) = (k_CH3OOH_OH);
        rconst(index,117) = (.5 *(1.36E-15 *exp(- 2112. / temp_loc)+ 7.51E-16 *exp(- 1521. / temp_loc)));
        rconst(index,118) = (.5 *(4.1E-12 *exp(452. / temp_loc)+ 1.9E-11 *exp(175. / temp_loc)));
        rconst(index,119) = (1.82E-13 *exp(1300. / temp_loc));
        rconst(index,120) = (2.54E-12 *exp(360. / temp_loc));
        rconst(index,121) = (.25 *k_3rd(temp_loc , cair_loc , 9.7E-29 , 5.6 , 9.3E-12 , 1.5 , 0.6));
        rconst(index,125) = (1.3E-12 *exp(- 25. / temp_loc));
        rconst(index,126) = (k_PrO2_HO2);
        rconst(index,127) = (k_PrO2_NO);
        rconst(index,128) = (k_CH3OOH_OH);
        rconst(index,131) = (k_PAN_M);
        rconst(index,132) = (7.86E-15 *exp(- 1913. / temp_loc));
        rconst(index,133) = (2.54E-11 *exp(410. / temp_loc));
        rconst(index,134) = (3.03E-12 *exp(- 446. / temp_loc));
        rconst(index,135) = (2.22E-13 *exp(1300. / temp_loc));
        rconst(index,136) = (2.54E-12 *exp(360. / temp_loc));
        rconst(index,141) = (2.8E-11 *exp(- 250. / temp_loc));
        rconst(index,142) = (2.5E-11 *exp(110. / temp_loc));
        rconst(index,143) = (1.0E-12 *exp(- 1590. / temp_loc));
        rconst(index,144) = (3.0E-11 *exp(- 2450. / temp_loc));
        rconst(index,145) = (3.5E-13 *exp(- 1370. / temp_loc));
        rconst(index,146) = (k_ClO_ClO);
        rconst(index,147) = (k_ClO_ClO / (2.16E-27 *exp(8537. / temp_loc)));
        rconst(index,148) = (3.9E-11 *exp(- 2310. / temp_loc));
        rconst(index,149) = (4.4E-11- 7.5E-11 *exp(- 620. / temp_loc));
        rconst(index,150) = (7.5E-11 *exp(- 620. / temp_loc));
        rconst(index,151) = (1.1E-11 *exp(- 980. / temp_loc));
        rconst(index,152) = (7.3E-12 *exp(300. / temp_loc));
        rconst(index,153) = (2.2E-12 *exp(340. / temp_loc));
        rconst(index,154) = (1.7E-12 *exp(- 230. / temp_loc));
        rconst(index,155) = (3.0E-12 *exp(- 500. / temp_loc));
        rconst(index,156) = (6.2E-12 *exp(295. / temp_loc));
        rconst(index,157) = (k_3rd_iupac(temp_loc , cair_loc , 1.6E-31 , 3.4 , 7.E-11 , 0. , 0.4));
        rconst(index,158) = (6.918E-7 *exp(- 10909. / temp_loc) *cair_loc);
        rconst(index,159) = (4.5E-12 *exp(- 900. / temp_loc));
        rconst(index,160) = (6.2E-12 *exp(145. / temp_loc));
        rconst(index,161) = (6.6E-12 *exp(- 1240. / temp_loc));
        rconst(index,162) = (8.1E-11 *exp(- 34. / temp_loc));
        rconst(index,164) = (1.8E-12 *exp(- 600. / temp_loc));
        rconst(index,167) = (1.96E-12 *exp(- 1200. / temp_loc));
        rconst(index,169) = (1.64E-12 *exp(- 1520. / temp_loc));
        rconst(index,170) = (k_3rd_iupac(temp_loc , cair_loc , 1.85E-29 , 3.3 , 6.0E-10 , 0.0 , 0.4));
        rconst(index,174) = (1.7E-11 *exp(- 800. / temp_loc));
        rconst(index,175) = (1.9E-11 *exp(230. / temp_loc));
        rconst(index,177) = (2.9E-14 *exp(840. / temp_loc));
        rconst(index,178) = (7.7E-12 *exp(- 450. / temp_loc));
        rconst(index,179) = (4.5E-12 *exp(500. / temp_loc));
        rconst(index,180) = (6.7E-12 *exp(155. / temp_loc));
        rconst(index,181) = (1.2E-10 *exp(- 430. / temp_loc));
        rconst(index,182) = (2.0E-11 *exp(240. / temp_loc));
        rconst(index,184) = (8.7E-12 *exp(260. / temp_loc));
        rconst(index,185) = (k_BrO_NO2);
        rconst(index,186) = (k_BrO_NO2 / (5.44E-9 *exp(14192. / temp_loc) *1.E6 *R_gas *temp_loc / (atm2Pa *N_A)));
        rconst(index,187) = (7.7E-12 *exp(- 580. / temp_loc));
        rconst(index,188) = (2.6E-12 *exp(- 1600. / temp_loc));
        rconst(index,189) = (G7402a_yield *5.7E-12);
        rconst(index,190) = ((1.- G7402a_yield) *5.7E-12);
        rconst(index,191) = (1.42E-12 *exp(- 1150. / temp_loc));
        rconst(index,192) = (2.8E-13 *exp(224. / temp_loc) / (1.+ 1.13E24 *exp(- 3200. / temp_loc) / var(index,ind_O2)));
        rconst(index,193) = (1.8e-11 *exp(- 460. / temp_loc));
        rconst(index,194) = (9.0E-13 *exp(- 360. / temp_loc));
        rconst(index,195) = (2.0E-12 *exp(- 840. / temp_loc));
        rconst(index,198) = (2.3E-10 *exp(135. / temp_loc));
        rconst(index,199) = (1.6E-12 *exp(430. / temp_loc));
        rconst(index,200) = (2.9E-12 *exp(220. / temp_loc));
        rconst(index,201) = (5.8E-13 *exp(170. / temp_loc));
        rconst(index,203) = (2.0E-12 *exp(- 840. / temp_loc));
        rconst(index,204) = (2.0E-12 *exp(- 840. / temp_loc));
        rconst(index,205) = (2.1E-12 *exp(- 880. / temp_loc));
        rconst(index,206) = (k_3rd(temp_loc , cair_loc , 3.3E-31 , 4.3 , 1.6E-12 , 0. , 0.6));
        rconst(index,207) = (1.13E-11 *exp(- 253. / temp_loc));
        rconst(index,208) = (k_DMS_OH);
        rconst(index,209) = (1.9E-13 *exp(520. / temp_loc));
        rconst(index,211) = (1.8E13 *exp(- 8661. / temp_loc));
        rconst(index,215) = (9.E-11 *exp(- 2386. / temp_loc));
        rconst(index,217) = (jx(index,ip_O2));
        rconst(index,218) = (jx(index,ip_O1D));
        rconst(index,219) = (jx(index,ip_O3P));
        rconst(index,220) = (jx(index,ip_H2O));
        rconst(index,221) = (jx(index,ip_H2O2));
        rconst(index,222) = (jx(index,ip_N2O));
        rconst(index,223) = (jx(index,ip_NO2));
        rconst(index,224) = (jx(index,ip_NO));
        rconst(index,225) = (jx(index,ip_NO2O));
        rconst(index,226) = (jx(index,ip_NOO2));
        rconst(index,227) = (jx(index,ip_N2O5));
        rconst(index,228) = (jx(index,ip_HONO));
        rconst(index,229) = (jx(index,ip_HNO3));
        rconst(index,230) = (jx(index,ip_HNO4));
        rconst(index,231) = (jx(index,ip_CH3OOH));
        rconst(index,232) = (jx(index,ip_COH2));
        rconst(index,233) = (jx(index,ip_CHOH));
        rconst(index,234) = (jx(index,ip_CO2));
        rconst(index,235) = (jx(index,ip_CH4));
        rconst(index,236) = (jx(index,ip_CH3OOH));
        rconst(index,237) = (jx(index,ip_CH3CHO));
        rconst(index,238) = (jx(index,ip_CH3CO3H));
        rconst(index,239) = (0.19 *jx(index,ip_CHOH));
        rconst(index,240) = (jx(index,ip_PAN));
        rconst(index,241) = (jx(index,ip_CH3OOH));
        rconst(index,242) = (jx(index,ip_CH3COCH3));
        rconst(index,243) = (0.074 *jx(index,ip_CHOH));
        rconst(index,244) = (jx(index,ip_MGLYOX));
        rconst(index,245) = (jx(index,ip_CH3OOH));
        rconst(index,246) = (3.7 *jx(index,ip_PAN));
        rconst(index,247) = (jx(index,ip_CH3OOH));
        rconst(index,248) = (0.019 *jx(index,ip_COH2)+ .015 *jx(index,ip_MGLYOX));
        rconst(index,249) = (jx(index,ip_CH3OOH));
        rconst(index,250) = (0.42 *jx(index,ip_CHOH));
        rconst(index,251) = (jx(index,ip_CH3OOH));
        rconst(index,252) = (2.15 *jx(index,ip_MGLYOX));
        rconst(index,253) = (3.7 *jx(index,ip_PAN));
        rconst(index,254) = (jx(index,ip_PAN));
        rconst(index,255) = (jx(index,ip_CH3OOH));
        rconst(index,256) = (3.7 *jx(index,ip_PAN));
        rconst(index,257) = (jx(index,ip_Cl2));
        rconst(index,258) = (jx(index,ip_Cl2O2));
        rconst(index,259) = (jx(index,ip_OClO));
        rconst(index,260) = (jx(index,ip_HCl));
        rconst(index,261) = (jx(index,ip_HOCl));
        rconst(index,262) = (jx(index,ip_ClNO2));
        rconst(index,263) = (jx(index,ip_ClNO3));
        rconst(index,264) = (jx(index,ip_ClONO2));
        rconst(index,265) = (jx(index,ip_CH3Cl));
        rconst(index,266) = (jx(index,ip_CCl4));
        rconst(index,267) = (jx(index,ip_CH3CCl3));
        rconst(index,268) = (jx(index,ip_CFCl3));
        rconst(index,269) = (jx(index,ip_CF2Cl2));
        rconst(index,270) = (jx(index,ip_Br2));
        rconst(index,271) = (jx(index,ip_BrO));
        rconst(index,272) = (jx(index,ip_HOBr));
        rconst(index,273) = (jx(index,ip_BrNO2));
        rconst(index,274) = (jx(index,ip_BrNO3));
        rconst(index,275) = (jx(index,ip_CH3Br));
        rconst(index,276) = (jx(index,ip_CH2Br2));
        rconst(index,277) = (jx(index,ip_CHBr3));
        rconst(index,278) = (jx(index,ip_CF3Br));
        rconst(index,279) = (jx(index,ip_BrCl));
        rconst(index,280) = (jx(index,ip_CF2ClBr));
        rconst(index,281) = (jx(index,ip_CH2ClBr));
        rconst(index,282) = (jx(index,ip_CHCl2Br));
        rconst(index,283) = (jx(index,ip_CHClBr2));
        rconst(index,284) = (jx(index,ip_CH3I));
        rconst(index,285) = (khet_st(index,ihs_N2O5_H2O));
        rconst(index,286) = (khet_tr(index,iht_N2O5));
        rconst(index,287) = (khet_st(index,ihs_HOCl_HCl));
        rconst(index,288) = (khet_st(index,ihs_ClNO3_HCl));
        rconst(index,289) = (khet_st(index,ihs_ClNO3_H2O));
        rconst(index,290) = (khet_st(index,ihs_N2O5_HCl));
        rconst(index,291) = (khet_st(index,ihs_HOBr_HBr));
        rconst(index,292) = (khet_st(index,ihs_BrNO3_H2O));
        rconst(index,293) = (khet_st(index,ihs_ClNO3_HBr));
        rconst(index,294) = (khet_st(index,ihs_BrNO3_HCl));
        rconst(index,295) = (khet_st(index,ihs_HOCl_HBr));
        rconst(index,296) = (khet_st(index,ihs_HOBr_HCl));
        rconst(index,297) = (k_O3s);
        rconst(index,299) = (jx(index,ip_CFCl3));
        rconst(index,301) = (jx(index,ip_CF2Cl2));
        rconst(index,302) = (7.25E-11 *exp(20. / temp_loc));
        rconst(index,303) = (4.63E-11 *exp(20. / temp_loc));
        rconst(index,304) = (jx(index,ip_N2O));
        rconst(index,306) = (1.64E-12 *exp(- 1520. / temp_loc));
        rconst(index,307) = (jx(index,ip_CH3CCl3));
        rconst(index,308) = (jx(index,ip_CF2ClBr));
        rconst(index,309) = (jx(index,ip_CF3Br));
        rconst(index,(3)-1) = 1.2e-10;
        rconst(index,(7)-1) = 1.2e-10;
        rconst(index,(13)-1) = 7.2e-11;
        rconst(index,(14)-1) = 6.9e-12;
        rconst(index,(15)-1) = 1.6e-12;
        rconst(index,(19)-1) = 1.8e-12;
        rconst(index,(36)-1) = 3.5e-12;
        rconst(index,(49)-1) = 1.2e-14;
        rconst(index,(50)-1) = 1300;
        rconst(index,(54)-1) = 1.66e-12;
        rconst(index,(57)-1) = 1.75e-10;
        rconst(index,(62)-1) = 1.3e-12;
        rconst(index,(69)-1) = 4e-13;
        rconst(index,(75)-1) = 2.3e-12;
        rconst(index,(85)-1) = 4e-12;
        rconst(index,(123)-1) = 2e-12;
        rconst(index,(124)-1) = 2e-12;
        rconst(index,(125)-1) = 3e-11;
        rconst(index,(130)-1) = 1.7e-12;
        rconst(index,(131)-1) = 3.2e-11;
        rconst(index,(138)-1) = 2e-12;
        rconst(index,(139)-1) = 2e-12;
        rconst(index,(140)-1) = 1e-10;
        rconst(index,(141)-1) = 1.3e-11;
        rconst(index,(164)-1) = 5.9e-11;
        rconst(index,(166)-1) = 3.3e-10;
        rconst(index,(167)-1) = 1.65e-10;
        rconst(index,(169)-1) = 3.25e-10;
        rconst(index,(172)-1) = 8e-11;
        rconst(index,(173)-1) = 1.4e-10;
        rconst(index,(174)-1) = 2.3e-10;
        rconst(index,(177)-1) = 2.7e-12;
        rconst(index,(184)-1) = 4.9e-11;
        rconst(index,(197)-1) = 3.32e-15;
        rconst(index,(198)-1) = 1.1e-15;
        rconst(index,(203)-1) = 1.45e-11;
        rconst(index,(211)-1) = 1e-10;
        rconst(index,(213)-1) = 3e-13;
        rconst(index,(214)-1) = 5e-11;
        rconst(index,(215)-1) = 3.3e-10;
        rconst(index,(217)-1) = 4.4e-13;
        rconst(index,(299)-1) = 2.3e-10;
        rconst(index,(301)-1) = 1.4e-10;
        rconst(index,(306)-1) = 3e-10;
    }
}


__global__ 
void Rosenbrock(double * __restrict__ conc, const double Tstart, const double Tend, double * __restrict__ rstatus, int * __restrict__ istatus,
                // values calculated from icntrl and rcntrl at host
                const int autonomous, const int vectorTol, const int UplimTol, const int method, const int Max_no_steps,
                const double Hmin, const double Hmax, const double Hstart, const double FacMin, const double FacMax, const double FacRej, const double FacSafe, const double roundoff,
                // cuda global mem buffers              
                const double * __restrict__ absTol, const double * __restrict__ relTol,
                // for update_rconst
    	        const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
		const double * __restrict__ jx,
                // global input
                const double * __restrict__ temp_gpu,
                const double * __restrict__ press_gpu,
                const double * __restrict__ cair_gpu,
                // extra
                const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    /* Temporary arrays allocated in stack */

    /* 
     *  Optimization NOTE: runs faster on Tesla/Fermi 
     *  when tempallocated on stack instead of heap.
     *  In theory someone can aggregate accesses together,
     *  however due to algorithm, threads access 
     *  different parts of memory, making it harder to
     *  optimize accesses. 
     *
     */
    double varNew_stack[NVAR];
    double var_stack[NSPEC];
    double varErr_stack[NVAR];
    double fix_stack[NFIX];
    double Fcn0_stack[NVAR];
    double jac0_stack[LU_NONZERO];
    double dFdT_stack[NVAR];
    double Ghimj_stack[LU_NONZERO];
    double K_stack[6*NVAR];


    /* Allocated in Global mem */
    double rconst_stack[NREACT];

    /* Allocated in stack */
    double *Ghimj  = Ghimj_stack;
    double *K      = K_stack;
    double *varNew = varNew_stack;
    double *Fcn0   = Fcn0_stack;
    double *dFdT   = dFdT_stack;
    double *jac0   = jac0_stack;
    double *varErr = varErr_stack;
    double *var    = var_stack;
    double *fix    = fix_stack;  
    double *rconst = rconst_stack;

    if (index < VL_GLO)
    {

        int Nfun,Njac,Nstp,Nacc,Nrej,Ndec,Nsol,Nsng;
        double Texit, Hexit;

        Nfun = 0;
        Njac = 0;
        Nstp = 0;
        Nacc = 0;
        Nrej = 0;
        Ndec = 0;
        Nsol = 0;
        Nsng = 0;

        /* FIXME: add check for method */
        const double *ros_A     = &ros[method-1].ros_A[0]; 
        const double *ros_C     = &ros[method-1].ros_C[0];
        const double *ros_M     = &ros[method-1].ros_M[0]; 
        const double *ros_E     = &ros[method-1].ros_E[0];
        const double *ros_Alpha = &ros[method-1].ros_Alpha[0]; 
        const double *ros_Gamma = &ros[method-1].ros_Gamma[0]; 
        const int    *ros_NewF  = &ros[method-1].ros_NewF[0];
        const int     ros_S     =  ros[method-1].ros_S; 
        const double  ros_ELO   =  ros[method-1].ros_ELO; 





        /* Copy data from global memory to temporary array */
        /*
         * Optimization note: if we ever have enough constant
         * memory, we could use it for storing the data.
         * In current architectures if we use constant memory
         * only a few threads will be able to run on the fly.
         *
         */
        for (int i=0; i<NSPEC; i++)
            var(index,i) = conc(index,i);

        for (int i=0; i<NFIX; i++)
            fix(index,i) = conc(index,NVAR+i);


        update_rconst(var, khet_st, khet_tr, jx, rconst, temp_gpu, press_gpu, cair_gpu, VL_GLO); 

        ros_Integrator(var, fix, Tstart, Tend, Texit,
                //  Rosenbrock method coefficients
                ros_S, ros_M, ros_E, ros_A, ros_C, 
                ros_Alpha, ros_Gamma, ros_ELO, ros_NewF, 
                //  Integration parameters
                autonomous, vectorTol, Max_no_steps, 
                roundoff, Hmin, Hmax, Hstart, Hexit, 
                FacMin, FacMax, FacRej, FacSafe,
                //  Status parameters
                Nfun, Njac, Nstp, Nacc, Nrej, Ndec, Nsol, Nsng,
                //  cuda global mem buffers              
                rconst, absTol, relTol, varNew, Fcn0,  
                K, dFdT, jac0, Ghimj,  varErr, 
                // For update rconst
                khet_st, khet_tr, jx,
                VL_GLO
                );

        for (int i=0; i<NVAR; i++)
            conc(index,i) = var(index,i); 


        /* Statistics */
        istatus(index,ifun) = Nfun;
        istatus(index,ijac) = Njac;
        istatus(index,istp) = Nstp;
        istatus(index,iacc) = Nacc;
        istatus(index,irej) = Nrej;
        istatus(index,idec) = Ndec;
        istatus(index,isol) = Nsol;
        istatus(index,isng) = Nsng;
        // Last T and H
        rstatus(index,itexit) = Texit;
        rstatus(index,ihexit) = Hexit; 
    }
}



__device__ static int ros_Integrator_ros3(double * __restrict__ var, const double * __restrict__ fix, const double Tstart, const double Tend, double &T,
        //  Integration parameters
        const int autonomous, const int vectorTol, const int Max_no_steps, 
        const double roundoff, const double Hmin, const double Hmax, const double Hstart, double &Hexit, 
        const double FacMin, const double FacMax, const double FacRej, const double FacSafe, 
        //  Status parameters
        int &Nfun, int &Njac, int &Nstp, int &Nacc, int &Nrej, int &Ndec, int &Nsol, int &Nsng,
        //  cuda global mem buffers              
        const double * __restrict__ rconst,  const double * __restrict__ absTol, const double * __restrict__ relTol, double * __restrict__ varNew, double * __restrict__ Fcn0, 
        double * __restrict__ K, double * __restrict__ dFdT, double * __restrict__ jac0, double * __restrict__ Ghimj, double * __restrict__ varErr,
        // for update_rconst
        const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
        const double * __restrict__ jx,
        // VL_GLO
        const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    double H, Hnew, HC, HC0,HC1, HG, Fac; // Tau - not used
    double Err; //*varErr;
    int direction;
    int rejectLastH, rejectMoreH;
    const double DELTAMIN = 1.0E-5;

    const int ros_S = 3;

    //   ~~~>  Initial preparations
    T = Tstart;
    Hexit = 0.0;
    H = fmin(Hstart,Hmax);
    if (fabs(H) <= 10.0*roundoff) 
        H = DELTAMIN;

    if (Tend  >=  Tstart)
    {
        direction = + 1;
    }
    else
    {
        direction = - 1;
    }

    rejectLastH=0;
    rejectMoreH=0;

    // TimeLoop: 
    while((direction > 0) && ((T- Tend)+ roundoff <= ZERO) || (direction < 0) && ((Tend-T)+ roundoff <= ZERO))
    {
        if (Nstp > Max_no_steps) //  Too many steps
            return -6;
        //  Step size too small
        if (H <= roundoff){  //  Step size too small
            //if (((T+ 0.1*H) == T) || (H <= roundoff)) {
            return -7;
        }

        //   ~~~>  Limit H if necessary to avoid going beyond Tend
        Hexit = H;
        H = fmin(H,fabs(Tend-T));

        //   ~~~>   Compute the function at current time
        Fun(var, fix, rconst, Fcn0, Nfun, VL_GLO);

        //   ~~~>  Compute the function derivative with respect to T
        if (!autonomous)
            ros_FunTimeDerivative(T, roundoff, var, fix, rconst, dFdT, Fcn0, Nfun, khet_st, khet_tr, jx,  VL_GLO); /// VAR READ - fcn0 read

        //   ~~~>   Compute the Jacobian at current time
        Jac_sp(var, fix, rconst, jac0, Njac, VL_GLO);   /// VAR READ 

        //   ~~~>  Repeat step calculation until current step accepted
        // UntilAccepted: 
        while(1)
        {
            ros_PrepareMatrix(H, direction, 0.43586652150845899941601945119356E+00 , jac0, Ghimj, Nsng, Ndec, VL_GLO);

            { // istage=0
                for (int i=0; i<NVAR; i++){
                    K(index,0,i)  = Fcn0(index,i);				// FCN0 Read
                }

                if ((!autonomous))
                {
                    HG = direction*H*0.43586652150845899941601945119356E+00;
                    for (int i=0; i<NVAR; i++){
                        K(index,0,i) += dFdT(index,i)*HG;
		     }
                }
                ros_Solve(Ghimj, K, Nsol, 0, ros_S);
            } // Stage

            {   // istage = 1
                for (int i=0; i<NVAR; i++){		
                    varNew(index,i) = K(index,0,i)  + var(index,i);
                }
                Fun(varNew, fix, rconst, varNew, Nfun,VL_GLO); // FCN <- varNew / not overlap 
                HC = -0.10156171083877702091975600115545E+01/(direction*H);
                for (int i=0; i<NVAR; i++){
                    double tmp = K(index,0,i);
                    K(index,1,i) = tmp*HC + varNew(index,i);
                }
                if ((!autonomous))
                {
                    HG = direction*H*0.24291996454816804366592249683314E+00;
                    for (int i=0; i<NVAR; i++){
                        K(index,1,i) += dFdT(index,i)*HG;
		     }
                }
		//	   R   ,RW, RW,  R,        R 
                ros_Solve(Ghimj, K, Nsol, 1, ros_S);
            } // Stage

            {
                int istage = 2;

                HC0 = 0.40759956452537699824805835358067E+01/(direction*H);
                HC1 = 0.92076794298330791242156818474003E+01/(direction*H);

                for (int i=0; i<NVAR; i++){
                    K(index,2,i) = K(index,1,i)*HC1 +   K(index,0,i)*HC0 +  varNew(index,i);
                }
                if ((!autonomous) )
                {
                    HG = direction*H*0.21851380027664058511513169485832E+01;
                    for (int i=0; i<NVAR; i++){
                        K(index,istage,i) += dFdT(index,i)*HG;
		     }
                }
                ros_Solve(Ghimj, K, Nsol, istage, ros_S);
            } // Stage

            //  ~~~>  Compute the new solution
	    for (int i=0; i<NVAR; i++){
                    varNew(index,i) = K(index,0,i)   + K(index,1,i)*0.61697947043828245592553615689730E+01 + K(index,2,i)*(-0.42772256543218573326238373806514) + var(index,i) ;
                    varErr(index,i) = K(index,0,i)/2 + K(index,1,i)*(-0.29079558716805469821718236208017E+01) + K(index,2,i)*(0.22354069897811569627360909276199);
	    }

            Err = ros_ErrorNorm(var, varNew, varErr, absTol, relTol, vectorTol);   

//  ~~~> New step size is bounded by FacMin <= Hnew/H <= FacMax
            Fac  = fmin(FacMax,fmax(FacMin,FacSafe/pow(Err,ONE/3.0)));
            Hnew = H*Fac;

//  ~~~>  Check the error magnitude and adjust step size
            Nstp = Nstp+ 1;
            if((Err <= ONE) || (H <= Hmin)) // ~~~> Accept step
            {
                Nacc = Nacc + 1;
                for (int j=0; j<NVAR ; j++)
                    var(index,j) =  fmax(varNew(index,j),ZERO);  /////////// VAR WRITE - last VarNew read

                T = T +  direction*H;
                Hnew = fmax(Hmin,fmin(Hnew,Hmax));
                if (rejectLastH)   // No step size increase after a rejected step
                    Hnew = fmin(Hnew,H);
                rejectLastH = 0;
                rejectMoreH = 0;
                H = Hnew;

            	break;  //  EXIT THE LOOP: WHILE STEP NOT ACCEPTED
            }
            else      // ~~~> Reject step
            {
                if (rejectMoreH)
                    Hnew = H*FacRej;
                rejectMoreH = rejectLastH;
                rejectLastH = 1;
                H = Hnew;
                if (Nacc >= 1)
                    Nrej += 1;
            } //  Err <= 1
        } // UntilAccepted
    } // TimeLoop
//  ~~~> Succesful exit
    return 0; //  ~~~> The integration was successful
}

__global__ 
void Rosenbrock_ros3(double * __restrict__ conc, const double Tstart, const double Tend, double * __restrict__ rstatus, int * __restrict__ istatus,
                const int autonomous, const int vectorTol, const int UplimTol, const int Max_no_steps,
                const double Hmin, const double Hmax, const double Hstart, const double FacMin, const double FacMax, const double FacRej, const double FacSafe, const double roundoff,
                const double * __restrict__ absTol, const double * __restrict__ relTol,
    	        const double * __restrict__ khet_st, const double * __restrict__ khet_tr,
		const double * __restrict__ jx,
                const double * __restrict__ temp_gpu,
                const double * __restrict__ press_gpu,
                const double * __restrict__ cair_gpu,
                const int VL_GLO)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;

    /* Temporary arrays allocated in stack */

    /* 
     *  Optimization NOTE: runs faster on Tesla/Fermi 
     *  when tempallocated on stack instead of heap.
     *  In theory someone can aggregate accesses together,
     *  however due to algorithm, threads access 
     *  different parts of memory, making it harder to
     *  optimize accesses. 
     *
     */
    double varNew_stack[NVAR];
    double var_stack[NVAR];
    double varErr_stack[NVAR];
    double fix_stack[NFIX];
    double Fcn0_stack[NVAR];
    double jac0_stack[LU_NONZERO];
    double dFdT_stack[NVAR];
    double Ghimj_stack[LU_NONZERO];
    double K_stack[3*NVAR];
    double rconst_stack[NREACT];

    /* Allocated in stack */
    double *Ghimj  = Ghimj_stack;
    double *K      = K_stack;
    double *varNew = varNew_stack;
    double *Fcn0   = Fcn0_stack;
    double *dFdT   = dFdT_stack;
    double *jac0   = jac0_stack;
    double *varErr = varErr_stack;
    double *var    = var_stack;
    double *fix    = fix_stack;  
    double *rconst = rconst_stack;

    const int method = 2;

    if (index < VL_GLO)
    {

        int Nfun,Njac,Nstp,Nacc,Nrej,Ndec,Nsol,Nsng;
        double Texit, Hexit;

        Nfun = 0;
        Njac = 0;
        Nstp = 0;
        Nacc = 0;
        Nrej = 0;
        Ndec = 0;
        Nsol = 0;
        Nsng = 0;



        /* Copy data from global memory to temporary array */
        /*
         * Optimization note: if we ever have enough constant
         * memory, we could use it for storing the data.
         * In current architectures if we use constant memory
         * only a few threads will be able to run on the fly.
         *
         */
        for (int i=0; i<NSPEC; i++)
            var(index,i) = conc(index,i);

        for (int i=0; i<NFIX; i++)
            fix(index,i) = conc(index,NVAR+i);

        //update_rconst(var, khet_st, khet_tr, jx, VL_GLO);
        update_rconst(var, khet_st, khet_tr, jx, rconst, temp_gpu, press_gpu, cair_gpu, VL_GLO); 

        ros_Integrator_ros3(var, fix, Tstart, Tend, Texit,
                //  Integration parameters
                autonomous, vectorTol, Max_no_steps, 
                roundoff, Hmin, Hmax, Hstart, Hexit, 
                FacMin, FacMax, FacRej, FacSafe,
                //  Status parameters
                Nfun, Njac, Nstp, Nacc, Nrej, Ndec, Nsol, Nsng,
                //  cuda global mem buffers              
                rconst, absTol, relTol, varNew, Fcn0,  
                K, dFdT, jac0, Ghimj,  varErr, 
                // For update rconst
                khet_st, khet_tr, jx,
                VL_GLO
                );

        for (int i=0; i<NVAR; i++)
            conc(index,i) = var(index,i); 


        /* Statistics */
        istatus(index,ifun) = Nfun;
        istatus(index,ijac) = Njac;
        istatus(index,istp) = Nstp;
        istatus(index,iacc) = Nacc;
        istatus(index,irej) = Nrej;
        istatus(index,idec) = Ndec;
        istatus(index,isol) = Nsol;
        istatus(index,isng) = Nsng;
        // Last T and H
        rstatus(index,itexit) = Texit;
        rstatus(index,ihexit) = Hexit; 
    }
}






                                                        // no int8 in CUDA :(
__global__ void reduce_istatus_1(int *istatus, int4 *tmp_out_1, int4 *tmp_out_2, int VL_GLO, int *xNacc, int *xNrej)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int idx_1 = threadIdx.x;
    int global_size = blockDim.x*gridDim.x;
    
    int foo;
    //no int8 in CUDA :(
    int4 accumulator_1 = make_int4(0,0,0,0);
    int4 accumulator_2 = make_int4(0,0,0,0);
    while (index < VL_GLO)
    {
        accumulator_1.x += istatus(index,0);
        accumulator_1.y += istatus(index,1);
        accumulator_1.z += istatus(index,2);
        //some dirty work on the side...
        foo = istatus(index,3);
        xNacc[index] = foo;
        accumulator_1.w += foo;
        foo = istatus(index,4);
        xNrej[index] = foo;
        accumulator_2.x += foo;
        accumulator_2.y += istatus(index,5);
        accumulator_2.z += istatus(index,6);
        accumulator_2.w += istatus(index,7);
        index += global_size;
    }
    //no int8 in CUDA :(
    __shared__ int4 buffer_1[REDUCTION_SIZE_1];
    __shared__ int4 buffer_2[REDUCTION_SIZE_1];
    
    buffer_1[idx_1] = accumulator_1;
    buffer_2[idx_1] = accumulator_2;
    __syncthreads();
    
    int idx_2, active_threads = blockDim.x;
    int4 tmp_1, tmp_2;
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (idx_1 < active_threads)
        {
            idx_2 = idx_1+active_threads;
            
            tmp_1 = buffer_1[idx_1];
            tmp_2 = buffer_1[idx_2];
            
            tmp_1.x += tmp_2.x;
            tmp_1.y += tmp_2.y;
            tmp_1.z += tmp_2.z;
            tmp_1.w += tmp_2.w;
            
            buffer_1[idx_1] = tmp_1;
            
            
            tmp_1 = buffer_2[idx_1];
            tmp_2 = buffer_2[idx_2];
            
            tmp_1.x += tmp_2.x;
            tmp_1.y += tmp_2.y;
            tmp_1.z += tmp_2.z;
            tmp_1.w += tmp_2.w;
            
            buffer_2[idx_1] = tmp_1;
            
        }
        __syncthreads();
    }
    if (idx_1 == 0)
    {
        tmp_out_1[blockIdx.x] = buffer_1[0];
        tmp_out_2[blockIdx.x] = buffer_2[0];
    }
}            

__global__ void reduce_istatus_2(int4 *tmp_out_1, int4 *tmp_out_2, int *out)
{
    int idx_1 = threadIdx.x;
    //no int8 in CUDA :(
    __shared__ int4 buffer_1[REDUCTION_SIZE_2];
    __shared__ int4 buffer_2[REDUCTION_SIZE_2];
    
    buffer_1[idx_1] = tmp_out_1[idx_1];
    buffer_2[idx_1] = tmp_out_2[idx_1]; 
    __syncthreads();
    
    int idx_2, active_threads = blockDim.x;
    int4 tmp_1, tmp_2;
    while (active_threads != 1)
    {
        active_threads /= 2;
        if (idx_1 < active_threads)
        {
            idx_2 = idx_1+active_threads;
            
            tmp_1 = buffer_1[idx_1];
            tmp_2 = buffer_1[idx_2];
            
            tmp_1.x += tmp_2.x;
            tmp_1.y += tmp_2.y;
            tmp_1.z += tmp_2.z;
            tmp_1.w += tmp_2.w;
            
            buffer_1[idx_1] = tmp_1;
            
            
            tmp_1 = buffer_2[idx_1];
            tmp_2 = buffer_2[idx_2];
            
            tmp_1.x += tmp_2.x;
            tmp_1.y += tmp_2.y;
            tmp_1.z += tmp_2.z;
            tmp_1.w += tmp_2.w;
            
            buffer_2[idx_1] = tmp_1;
            
        }
        __syncthreads();
    }
    if (idx_1 == 0)
    {
        tmp_1 = buffer_1[0];
        tmp_2 = buffer_2[0];
        out[0] = tmp_1.x;
        out[1] = tmp_1.y;
        out[2] = tmp_1.z;
        out[3] = tmp_1.w;
        out[4] = tmp_2.x;
        out[5] = tmp_2.y;
        out[6] = tmp_2.z;
        out[7] = tmp_2.w;        
    }
}            

/* Assuming different processes */
enum { TRUE=1, FALSE=0 } ;
double *d_conc, *d_temp, *d_press, *d_cair, *d_khet_st, *d_khet_tr, *d_jx;
int initialized = FALSE;

/* Device pointers pointing to GPU */
double *d_rstatus, *d_absTol, *d_relTol;
int *d_istatus, *d_istatus_rd, *d_xNacc, *d_xNrej;
int4 *d_tmp_out_1, *d_tmp_out_2;

/* Allocate arrays on device for Rosenbrock */
__host__ void init_first_time(int pe, int VL_GLO, int size_khet_st, int size_khet_tr, int size_jx ){

    /* Select the proper GPU CARD */
    int deviceCount, device;
    gpuErrchk( cudaGetDeviceCount(&deviceCount) );
    device = pe % deviceCount;
    gpuErrchk( cudaSetDevice(device) );

    printf("PE[%d]: selected %d of total %d\n",pe,device,deviceCount);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1); 

    gpuErrchk( cudaMalloc ((void **) &d_conc   , sizeof(double)*VL_GLO*(NSPEC))        );
    gpuErrchk( cudaMalloc ((void **) &d_khet_st, sizeof(double)*VL_GLO*size_khet_st) );
    gpuErrchk( cudaMalloc ((void **) &d_khet_tr, sizeof(double)*VL_GLO*size_khet_tr) );
    gpuErrchk( cudaMalloc ((void **) &d_jx     , sizeof(double)*VL_GLO*size_jx)      );
   
    gpuErrchk( cudaMalloc ((void **) &d_rstatus    , sizeof(double)*VL_GLO*2)          );
    gpuErrchk( cudaMalloc ((void **) &d_istatus    , sizeof(int)*VL_GLO*8)             );
    gpuErrchk( cudaMalloc ((void **) &d_absTol     , sizeof(double)*NVAR)              );
    gpuErrchk( cudaMalloc ((void **) &d_relTol     , sizeof(double)*NVAR)              );

    /* Allocate input arrays */
    gpuErrchk( cudaMalloc ((void **) &temp_gpu     , sizeof(double)*VL_GLO)              );
    gpuErrchk( cudaMalloc ((void **) &press_gpu     , sizeof(double)*VL_GLO)              );
    gpuErrchk( cudaMalloc ((void **) &cair_gpu     , sizeof(double)*VL_GLO)              );

    /* Allocate arrays on device for reduce_foo */
    gpuErrchk( cudaMalloc ((void **) &d_istatus_rd  , sizeof(int)*8));
    gpuErrchk( cudaMalloc ((void **) &d_tmp_out_1   , sizeof(int4)*64));
    gpuErrchk( cudaMalloc ((void **) &d_tmp_out_2   , sizeof(int4)*64));
    gpuErrchk( cudaMalloc ((void **) &d_xNacc   , sizeof(int)*VL_GLO));
    gpuErrchk( cudaMalloc ((void **) &d_xNrej   , sizeof(int)*VL_GLO));
    

    initialized = TRUE;
}

/*
 * TODO: We should call it in some point..
 */
extern "C" void finalize_cuda(){
    /* Free memory on the device */
    gpuErrchk( cudaFree(d_conc        ) );
    gpuErrchk( cudaFree(d_temp        ) );
    gpuErrchk( cudaFree(d_press       ) );
    gpuErrchk( cudaFree(d_cair        ) );
    gpuErrchk( cudaFree(d_khet_st     ) );
    gpuErrchk( cudaFree(d_khet_tr     ) );
    gpuErrchk( cudaFree(d_jx          ) );
    gpuErrchk( cudaFree(d_rstatus     ) );
    gpuErrchk( cudaFree(d_istatus     ) );
    gpuErrchk( cudaFree(d_absTol      ) );
    gpuErrchk( cudaFree(d_relTol      ) );
    gpuErrchk( cudaFree(d_istatus_rd  ) ); 
    gpuErrchk( cudaFree(d_tmp_out_1   ) ); 
    gpuErrchk( cudaFree(d_tmp_out_2   ) ); 
    gpuErrchk( cudaFree(d_xNacc       ) ); 
    gpuErrchk( cudaFree(d_xNrej       ) ); 
    gpuErrchk( cudaFree(temp_gpu      ) ); 
    gpuErrchk( cudaFree(press_gpu     ) ); 
    gpuErrchk( cudaFree(cair_gpu      ) ); 
}



extern "C" void kpp_integrate_cuda_( int *pe_p, int *sizes, double *time_step_len_p, double *conc, double *temp, double *press, double *cair, 
                                    double *khet_st, double *khet_tr, double *jx, double *absTol, double *relTol, int *ierr, int *istatus, 
                                    int *xNacc, int *xNrej, double *rndoff, int *icntrl=NULL, double *rcntrl=NULL
				    ) 
/*  // TODO
 *  Parameters:
 *          pe_p: scalar int - processor element
 *        VL_GLO: scalar int - size of the system
 *         NSPEC: scalar int - number of species
 *        NREACT: scalar int - number of reactions
 *          NVAR: scalar int - 
 *
 *  Input data:
 *          conc: 2D array of doubles - size: vl_glo x number of species
 *          temp: 1D array of doubles - size: vl_glo
 *         press: 1D array of doubles - size: vl_glo
 *          cair: 1D array of doubles - size: vl_glo
 *       khet_st: 2D array of doubles - size: vl_glo x number of species
 *       khet_tr: 2D array of doubles - size: vl_glo x number of species 
 *            jx: 2D array of doubles - size: vl_glo x number of species
 *        absTol: 1D array of doubles - size: number of species
 *        relTol: 1D array of doubles - size: number of species
 *     Control:
 *        icntrl: 1D array of ints   - size: 4
 *         sizes: 1D array of ints   - size: 4
 *        rcntrl: 1D array of doubles - size: 7
 * 
 * 
 */
{

    const double DELTAMIN = 1.0E-5;


    
    int VL_GLO       = sizes[0];
    int size_khet_st = sizes[1];
    int size_khet_tr = sizes[2];
    int size_jx      = sizes[3];
    double roundoff  = *rndoff; 
    
    double Tstart,Tend;
    Tstart = ZERO;
    Tend   =  *time_step_len_p;
    int pe = *pe_p;
    
    // variables from rcntrl and icntrl
    int autonomous, vectorTol, UplimTol, method, Max_no_steps;
    double Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe;
    
    //int rcntrl_bool = 0, icntrl_bool=0;
    if (rcntrl == NULL)
    {
        rcntrl = new double[7];
        for (int i=0; i < 7; i++)
            rcntrl[i] = 0.0;
    }
    if (icntrl == NULL)
    {
        icntrl = new int[4];
        for (int i=0; i < 4; i++)
            icntrl[i] = 0;
    }

    /* Allocate arrays on device for update_rconst kernel*/        
    if (initialized == FALSE)   init_first_time(pe, VL_GLO, size_khet_st, size_khet_tr, size_jx);

    /* Copy data from host memory to device memory */
    gpuErrchk( cudaMemcpy(d_conc   , conc   	, sizeof(double)*VL_GLO*NSPEC        , cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(temp_gpu   , temp   	, sizeof(double)*VL_GLO  , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(press_gpu  , press  	, sizeof(double)*VL_GLO  , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(cair_gpu   , cair   	, sizeof(double)*VL_GLO  , cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMemcpy(d_khet_st, khet_st	, sizeof(double)*VL_GLO*size_khet_st , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_khet_tr, khet_tr	, sizeof(double)*VL_GLO*size_khet_tr , cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_jx     , jx     	, sizeof(double)*VL_GLO*size_jx      , cudaMemcpyHostToDevice) );

    /* Copy arrays from host memory to device memory for Rosenbrock */    
    gpuErrchk( cudaMemcpy(d_absTol, absTol, sizeof(double)*NVAR, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_relTol, relTol, sizeof(double)*NVAR, cudaMemcpyHostToDevice) );


    /* Compute execution configuration for update_rconst */
    int block_size, grid_size;
    
    block_size = BLOCKSIZE;
    grid_size = (VL_GLO + block_size - 1)/block_size;  
    dim3 dimBlock(block_size);
    dim3 dimGrid(grid_size);


    /* Execute the kernel */
    //update_rconst<<<dimGrid,dimBlock>>>(d_conc, d_khet_st, d_khet_tr, d_jx, VL_GLO); 

    GPU_DEBUG();
 
//  *------------------------------------------------------*
//  |    Default values vs input settings (icntrl, rcntrl) |
//  *------------------------------------------------------*
    int ierr_tmp=0;
    {
    //  autonomous or time dependent ODE. Default is time dependent.
        autonomous = !(icntrl[0] == 0);

    //  For Scalar tolerances (icntrl[1].NE.0)  the code uses absTol(0) and relTol(0)
    //  For Vector tolerances (icntrl[1] == 0) the code uses absTol(0:NVAR) and relTol(0:NVAR)
        if (icntrl[1] == 0)
        {
            vectorTol = 1; //bool
            UplimTol  = NVAR;
        }
        else
        {
            vectorTol = 0;
            UplimTol  = 1;
        }

    //  The particular Rosenbrock method chosen
        if (icntrl[2] == 0) 
        {
            method = 4;
        }
        else if ((icntrl[2] >= 1) && (icntrl[2] <= 5))
        {
            method = icntrl[2];
        }
        else
        {
            printf("User-selected Rosenbrock method: icntrl[2]=%d\n",method);
            ierr_tmp = -2;
        }
    //  The maximum number of steps admitted
        if (icntrl[3] == 0)
        {
            Max_no_steps = 100000;
        }
        else if (icntrl[3] > 0) 
        {
            Max_no_steps=icntrl[3];
        }
        else
        {
            printf("User-selected max no. of steps: icntrl[3]=%d\n",icntrl[3]);
            ierr_tmp = -1;
        }
    //  Unit roundoff (1+ roundoff>1)
        roundoff = machine_eps_flt(); 

    //  Lower bound on the step size: (positive value)
        if (rcntrl[0] == ZERO)
        {
            Hmin = ZERO;
        }
        else if (rcntrl[0] > ZERO) 
        {
            Hmin = rcntrl[0];
        }
        else
        {
            printf("User-selected Hmin: rcntrl[0]=%f\n",rcntrl[0]);
            ierr_tmp = -3;
        }
    //  Upper bound on the step size: (positive value)
        if (rcntrl[1] == ZERO) 
        {
            Hmax = fabs(Tend-Tstart);
        }
        else if (rcntrl[1] > ZERO) 
        {
            Hmax = fmin(fabs(rcntrl[1]),fabs(Tend-Tstart));
        }
        else
        {
            printf("User-selected Hmax: rcntrl[1]=%f\n",rcntrl[1]);
            ierr_tmp = -3;
        }
    //  Starting step size: (positive value)
        if (rcntrl[2] == ZERO) 
        {
            Hstart = fmax(Hmin,DELTAMIN);
        }
        else if (rcntrl[2] > ZERO) 
        {
            Hstart = fmin(fabs(rcntrl[2]),fabs(Tend-Tstart));
        }
        else
        {
            printf("User-selected Hstart: rcntrl[2]=%f\n",rcntrl[2]);
            ierr_tmp = -3;
        }
    //  Step size can be changed s.t.  FacMin < Hnew/Hexit < FacMax
        if (rcntrl[3] == ZERO)
        {
            FacMin = 0.2;
        }
        else if (rcntrl[3] > ZERO) 
        {
            FacMin = rcntrl[3];
        }
        else
        {
            printf("User-selected FacMin: rcntrl[3]=%f\n",rcntrl[3]);
            ierr_tmp = -4;
        }
        if (rcntrl[4] == ZERO) 
        {
            FacMax = 6.0;
        }
        else if (rcntrl[4] > ZERO) 
        {
            FacMax = rcntrl[4];
        }
        else
        {
            printf("User-selected FacMax: rcntrl[4]=%f\n",rcntrl[4]);
            ierr_tmp = -4; 
        }
    //  FacRej: Factor to decrease step after 2 succesive rejections
        if (rcntrl[5] == ZERO) 
        {
            FacRej = 0.1;
        }
        else if (rcntrl[5] > ZERO) 
        {
            FacRej = rcntrl[5];
        }
        else
        {
            printf("User-selected FacRej: rcntrl[5]=%f\n",rcntrl[5]);
            ierr_tmp = -4;
        }
    //  FacSafe: Safety Factor in the computation of new step size
        if (rcntrl[6] == ZERO) 
        {
            FacSafe = 0.9;
        }
        else if (rcntrl[6] > ZERO)
        {
            FacSafe = rcntrl[6];
        }
        else
        {
            printf("User-selected FacSafe: rcntrl[6]=%f\n",rcntrl[6]);
            ierr_tmp = -4;
        }
    //  Check if tolerances are reasonable
        for (int i=0; i < UplimTol; i++)
        {
            if ((absTol[i] <= ZERO) || (relTol[i] <= 10.0*roundoff) || (relTol[i] >= 1.0))
            {
                printf("CCC absTol(%d) = %f \n",i,absTol[i]);
                printf("CCC relTol(%d) = %f \n",i,relTol[i]);
                ierr_tmp = -5;
            }
        }
    }


      switch (method){
        case 2:
            Rosenbrock_ros3<<<dimGrid,dimBlock>>>(d_conc, Tstart, Tend, d_rstatus, d_istatus,
                    autonomous, vectorTol, UplimTol, Max_no_steps,
                    Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe, roundoff,
                    d_absTol, d_relTol,
                    d_khet_st, d_khet_tr, d_jx, 
                    temp_gpu, press_gpu, cair_gpu, 
                    VL_GLO);
            break;
        default: 
      Rosenbrock<<<dimGrid,dimBlock>>>(d_conc, Tstart, Tend, d_rstatus, d_istatus,
                    // values calculated from icntrl and rcntrl at host
                    autonomous, vectorTol, UplimTol, method, Max_no_steps,
                    Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe, roundoff,
                    //  cuda global mem buffers              
                    d_absTol, d_relTol,   
                    d_khet_st, d_khet_tr, d_jx, 
                    // Global input arrays
                    temp_gpu, press_gpu, cair_gpu, 
                    // extra - vector lenght and processor
                    VL_GLO); 
        
                    break;
    }

    GPU_DEBUG();

    
    reduce_istatus_1<<<REDUCTION_SIZE_2,REDUCTION_SIZE_1>>>(d_istatus, d_tmp_out_1, d_tmp_out_2, VL_GLO, d_xNacc, d_xNrej);


    GPU_DEBUG();

    reduce_istatus_2<<<1,REDUCTION_SIZE_2>>>(d_tmp_out_1, d_tmp_out_2, d_istatus_rd);

    GPU_DEBUG();
    
    /* Copy the result back */
    gpuErrchk( cudaMemcpy( conc      , d_conc       , sizeof(double)*VL_GLO*NVAR, cudaMemcpyDeviceToHost) );  
    gpuErrchk( cudaMemcpy( xNacc      , d_xNacc      , sizeof(int)*VL_GLO         , cudaMemcpyDeviceToHost) );  
    gpuErrchk( cudaMemcpy( xNrej      , d_xNrej      , sizeof(int)*VL_GLO         , cudaMemcpyDeviceToHost) ); 

    
    return;

}





