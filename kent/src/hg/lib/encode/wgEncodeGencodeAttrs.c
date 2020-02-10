/* wgEncodeGencodeAttrs.c was originally generated by the autoSql program, which also 
 * generated wgEncodeGencodeAttrs.h and wgEncodeGencodeAttrs.sql.  This module links the database and
 * the RAM representation of objects. */

#include "common.h"
#include "linefile.h"
#include "dystring.h"
#include "jksql.h"
#include "encode/wgEncodeGencodeAttrs.h"



char *wgEncodeGencodeAttrsCommaSepFieldNames = "geneId,geneName,geneType,geneStatus,transcriptId,transcriptName,transcriptType,transcriptStatus,havanaGeneId,havanaTranscriptId,ccdsId,level,transcriptClass,proteinId";

void wgEncodeGencodeAttrsStaticLoad(char **row, int numColumns, struct wgEncodeGencodeAttrs *ret)
/* Load a row from wgEncodeGencodeAttrs table into ret.  The contents of ret will
 * be replaced at the next call to this function. */
{

ret->geneId = row[0];
ret->geneName = row[1];
ret->geneType = row[2];
ret->geneStatus = row[3];
ret->transcriptId = row[4];
ret->transcriptName = row[5];
ret->transcriptType = row[6];
ret->transcriptStatus = row[7];
ret->havanaGeneId = row[8];
ret->havanaTranscriptId = row[9];
ret->ccdsId = row[10];
ret->level = sqlSigned(row[11]);
ret->transcriptClass = row[12];
if (numColumns > 13)
    ret->proteinId = row[13];
}

struct wgEncodeGencodeAttrs *wgEncodeGencodeAttrsLoad(char **row, int numColumns)
/* Load a wgEncodeGencodeAttrs from row fetched with select * from wgEncodeGencodeAttrs
 * from database.  Dispose of this with wgEncodeGencodeAttrsFree(). */
{
struct wgEncodeGencodeAttrs *ret;

AllocVar(ret);
ret->geneId = cloneString(row[0]);
ret->geneName = cloneString(row[1]);
ret->geneType = cloneString(row[2]);
ret->geneStatus = cloneString(row[3]);
ret->transcriptId = cloneString(row[4]);
ret->transcriptName = cloneString(row[5]);
ret->transcriptType = cloneString(row[6]);
ret->transcriptStatus = cloneString(row[7]);
ret->havanaGeneId = cloneString(row[8]);
ret->havanaTranscriptId = cloneString(row[9]);
ret->ccdsId = cloneString(row[10]);
ret->level = sqlSigned(row[11]);
ret->transcriptClass = cloneString(row[12]);
if (numColumns > 13)
    ret->proteinId = cloneString(row[13]);
return ret;
}

struct wgEncodeGencodeAttrs *wgEncodeGencodeAttrsLoadAll(char *fileName) 
/* Load all wgEncodeGencodeAttrs from a whitespace-separated file.
 * Dispose of this with wgEncodeGencodeAttrsFreeList(). */
{
struct wgEncodeGencodeAttrs *list = NULL, *el;
struct lineFile *lf = lineFileOpen(fileName, TRUE);
char *row[14];

while (lineFileRow(lf, row))
    {
    el = wgEncodeGencodeAttrsLoad(row, WGENCODEGENCODEATTRS_NUM_COLS);
    slAddHead(&list, el);
    }
lineFileClose(&lf);
slReverse(&list);
return list;
}

struct wgEncodeGencodeAttrs *wgEncodeGencodeAttrsLoadAllByChar(char *fileName, char chopper) 
/* Load all wgEncodeGencodeAttrs from a chopper separated file.
 * Dispose of this with wgEncodeGencodeAttrsFreeList(). */
{
struct wgEncodeGencodeAttrs *list = NULL, *el;
struct lineFile *lf = lineFileOpen(fileName, TRUE);
char *row[14];

while (lineFileNextCharRow(lf, chopper, row, ArraySize(row)))
    {
    el = wgEncodeGencodeAttrsLoad(row, WGENCODEGENCODEATTRS_NUM_COLS);
    slAddHead(&list, el);
    }
lineFileClose(&lf);
slReverse(&list);
return list;
}

struct wgEncodeGencodeAttrs *wgEncodeGencodeAttrsCommaIn(char **pS, struct wgEncodeGencodeAttrs *ret)
/* Create a wgEncodeGencodeAttrs out of a comma separated string. 
 * This will fill in ret if non-null, otherwise will
 * return a new wgEncodeGencodeAttrs */
{
char *s = *pS;

if (ret == NULL)
    AllocVar(ret);
ret->geneId = sqlStringComma(&s);
ret->geneName = sqlStringComma(&s);
ret->geneType = sqlStringComma(&s);
ret->geneStatus = sqlStringComma(&s);
ret->transcriptId = sqlStringComma(&s);
ret->transcriptName = sqlStringComma(&s);
ret->transcriptType = sqlStringComma(&s);
ret->transcriptStatus = sqlStringComma(&s);
ret->havanaGeneId = sqlStringComma(&s);
ret->havanaTranscriptId = sqlStringComma(&s);
ret->ccdsId = sqlStringComma(&s);
ret->level = sqlSignedComma(&s);
ret->transcriptClass = sqlStringComma(&s);
ret->proteinId = sqlStringComma(&s);
*pS = s;
return ret;
}

void wgEncodeGencodeAttrsFree(struct wgEncodeGencodeAttrs **pEl)
/* Free a single dynamically allocated wgEncodeGencodeAttrs such as created
 * with wgEncodeGencodeAttrsLoad(). */
{
struct wgEncodeGencodeAttrs *el;

if ((el = *pEl) == NULL) return;
freeMem(el->geneId);
freeMem(el->geneName);
freeMem(el->geneType);
freeMem(el->geneStatus);
freeMem(el->transcriptId);
freeMem(el->transcriptName);
freeMem(el->transcriptType);
freeMem(el->transcriptStatus);
freeMem(el->havanaGeneId);
freeMem(el->havanaTranscriptId);
freeMem(el->ccdsId);
freeMem(el->transcriptClass);
freeMem(el->proteinId);
freez(pEl);
}

void wgEncodeGencodeAttrsFreeList(struct wgEncodeGencodeAttrs **pList)
/* Free a list of dynamically allocated wgEncodeGencodeAttrs's */
{
struct wgEncodeGencodeAttrs *el, *next;

for (el = *pList; el != NULL; el = next)
    {
    next = el->next;
    wgEncodeGencodeAttrsFree(&el);
    }
*pList = NULL;
}

void wgEncodeGencodeAttrsOutput(struct wgEncodeGencodeAttrs *el, FILE *f, char sep, char lastSep) 
/* Print out wgEncodeGencodeAttrs.  Separate fields with sep. Follow last field with lastSep. */
{
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->geneId);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->geneName);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->geneType);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->geneStatus);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->transcriptId);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->transcriptName);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->transcriptType);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->transcriptStatus);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->havanaGeneId);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->havanaTranscriptId);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->ccdsId);
if (sep == ',') fputc('"',f);
fputc(sep,f);
fprintf(f, "%d", el->level);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->transcriptClass);
if (sep == ',') fputc('"',f);
fputc(sep,f);
if (sep == ',') fputc('"',f);
fprintf(f, "%s", el->proteinId);
if (sep == ',') fputc('"',f);
fputc(lastSep,f);
}

/* -------------------------------- End autoSql Generated Code -------------------------------- */
