/* hapmapAlleles.h was originally generated by the autoSql program, which also 
 * generated hapmapAlleles.c and hapmapAlleles.sql.  This header links the database and
 * the RAM representation of objects. */

/* Copyright (C) 2007 The Regents of the University of California 
 * See README in this or parent directory for licensing information. */

#ifndef HAPMAPALLELES_H
#define HAPMAPALLELES_H

#define HAPMAPALLELES_NUM_COLS 12

struct hapmapAlleles
/* HapMap allele counts */
    {
    struct hapmapAlleles *next;  /* Next in singly linked list. */
    char *chrom;	/* Chromosome */
    unsigned chromStart;	/* Start position in chrom (0 based) */
    unsigned chromEnd;	/* End position in chrom (1 based) */
    char *name;	/* Reference SNP identifier from dbSnp */
    unsigned score;	/* Not used */
    char strand[2];	/* Which genomic strand contains the observed alleles */
    char *observed;	/* Observed string from genotype file */
    char allele1[2];	/* This allele has been observed */
    unsigned homoCount1;	/* Count of individuals who are homozygous for allele1 */
    char *allele2;	/* This allele may not have been observed */
    unsigned homoCount2;	/* Count of individuals who are homozygous for allele2 */
    unsigned heteroCount;	/* Count of individuals who are heterozygous */
    };

void hapmapAllelesStaticLoad(char **row, struct hapmapAlleles *ret);
/* Load a row from hapmapAlleles table into ret.  The contents of ret will
 * be replaced at the next call to this function. */

struct hapmapAlleles *hapmapAllelesLoad(char **row);
/* Load a hapmapAlleles from row fetched with select * from hapmapAlleles
 * from database.  Dispose of this with hapmapAllelesFree(). */

struct hapmapAlleles *hapmapAllelesLoadAll(char *fileName);
/* Load all hapmapAlleles from whitespace-separated file.
 * Dispose of this with hapmapAllelesFreeList(). */

struct hapmapAlleles *hapmapAllelesLoadAllByChar(char *fileName, char chopper);
/* Load all hapmapAlleles from chopper separated file.
 * Dispose of this with hapmapAllelesFreeList(). */

#define hapmapAllelesLoadAllByTab(a) hapmapAllelesLoadAllByChar(a, '\t');
/* Load all hapmapAlleles from tab separated file.
 * Dispose of this with hapmapAllelesFreeList(). */

struct hapmapAlleles *hapmapAllelesCommaIn(char **pS, struct hapmapAlleles *ret);
/* Create a hapmapAlleles out of a comma separated string. 
 * This will fill in ret if non-null, otherwise will
 * return a new hapmapAlleles */

void hapmapAllelesFree(struct hapmapAlleles **pEl);
/* Free a single dynamically allocated hapmapAlleles such as created
 * with hapmapAllelesLoad(). */

void hapmapAllelesFreeList(struct hapmapAlleles **pList);
/* Free a list of dynamically allocated hapmapAlleles's */

void hapmapAllelesOutput(struct hapmapAlleles *el, FILE *f, char sep, char lastSep);
/* Print out hapmapAlleles.  Separate fields with sep. Follow last field with lastSep. */

#define hapmapAllelesTabOut(el,f) hapmapAllelesOutput(el,f,'\t','\n');
/* Print out hapmapAlleles as a line in a tab-separated file. */

#define hapmapAllelesCommaOut(el,f) hapmapAllelesOutput(el,f,',',',');
/* Print out hapmapAlleles as a comma separated list including final comma. */

/* -------------------------------- End autoSql Generated Code -------------------------------- */
/* items for trackUi options and filters */

#define HA_POP_MIXED "hapmapAlleles.isMixed"
#define HA_POP_COUNT "hapmapAlleles.popCount"
#define HA_TYPE "hapmapAlleles.polyType"
#define HA_MIN_FREQ "hapmapAlleles.minorAlleleFreqMinimum"
#define HA_MAX_FREQ "hapmapAlleles.minorAlleleFreqMaximum"
#define HA_MIN_HET "hapmapAlleles.hetMinimum"
#define HA_MAX_HET "hapmapAlleles.hetMaximum"
#define HA_MONO "hapmapAlleles.monomorphic"
#define HA_CHIMP "hapmapAlleles.chimp"
#define HA_CHIMP_QUAL "hapmapAlleles.chimpQualScore"
#define HA_MACAQUE "hapmapAlleles.macaque"
#define HA_MACAQUE_QUAL "hapmapAlleles.macaqueQualScore"

#define HA_POP_MIXED_DEFAULT "any"
#define HA_POP_COUNT_DEFAULT "any"
#define HA_TYPE_DEFAULT      "any"

#define HA_MIN_FREQ_DEFAULT  "0.0"
#define HA_MAX_FREQ_DEFAULT  "0.5"
#define HA_MIN_HET_DEFAULT   "0.0"
#define HA_MAX_HET_DEFAULT   "0.5"

#define HA_MONO_DEFAULT      "any"

#define HA_CHIMP_DEFAULT     "any"
#define HA_CHIMP_QUAL_DEFAULT "0"
#define HA_MACAQUE_DEFAULT   "any"
#define HA_MACAQUE_QUAL_DEFAULT "0"

#endif /* HAPMAPALLELES_H */

