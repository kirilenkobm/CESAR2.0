# CESAR 2.0

CESAR 2.0 (Coding Exon Structure Aware Realigner 2.0) is a method to realign coding exons or genes to DNA sequences using a Hidden Markov Model [1].

Compared to its predecessor [2], CESAR 2.0 is 77X times faster on average (132X times faster for large exons) and requires 30-times less memory. In addition, CESAR 2.0 improves the accuracy of the comparative gene annotation by two new features. First, CESAR 2.0 substantially improves the identification of splice sites that have shifted over a larger distance, which improves the accuracy of detecting the correct exon boundaries. 
Second, CESAR 2.0 provides a new gene mode that re-aligns entire genes at once. This mode is able to recognize complete intron deletions and will annotate larger joined exons that arose by intron deletion events. 



# Installation
Just call 

`make` 

to build CESAR 2.0. The binary is called `cesar`. A precompiled linux 64bit binary is precompiledBinary_x86_64/cesar

The code is commented in doxygen style.
To compile a doxygen documentation of this program at `doc/doxygen/index.html`, call 

`make doc` 

# Mini example of annotating 2 genes in one query genome
```
# get code 
git clone https://github.com/hillerlab/CESAR2.0/
# this directory contains the mini example input data
cd extra/miniExample

# run CESAR on both genes listed in twoGenes.gp.forCESAR
annotateGenesViaCESAR.pl POLR3K hg38_oryAfe1.bb twoGenes.gp.forCESAR hg38 oryAfe1 CESARoutput 2bitDir ../../tools -maxMemory 1
annotateGenesViaCESAR.pl SNRNP25 hg38_oryAfe1.bb twoGenes.gp.forCESAR hg38 oryAfe1 CESARoutput 2bitDir ../../tools -maxMemory 1

# combine the results into a genePred file
bed2GenePred.pl oryAfe1 CESARoutput oryAfe1.gp

# the result in oryAfe1.gp looks like
POLR3K	JH864469	-	38382	42006	38382	42006	3	38382,39767,41895	38510,39855,42006
SNRNP25	JH864469	+	42379	44193	42379	44193	5	42379,42799,43011,43524,44135	42466,42890,43117,43599,44193
```

# Example of annotating genes in several genomes

This example illustrates the usage of CESAR 2.0 to annotate human genes in 4 vertebrate genomes (mouse, aardvark, chicken, falcon). A generalized workflow is below. 

```
# test directory
mkdir CESARTest; cd CESARTest
# get code and data
git clone https://github.com/hillerlab/CESAR2.0/
wget -r -nH --cut-dirs=2 --reject "index.html*" https://bds.mpi-cbg.de/hillerlab/CESAR2.0_Example .
gzip -d multiz_5way.maf.gz

# define a few environment variables (see below for details)
export inputGenes=ensGene.gp
export reference=hg38
export twoBitDir=2bitDir
export alignment=multiz_5way.bb
export querySpecies=mm10,oryAfe1,galGal4,falChe1
export outputDir=CESARoutput
export resultsDir=geneAnnotation
# this will set the maxMemory to the amount of RAM in your machine - 1 Gb
export maxMemory=`grep MemTotal /proc/meminfo | awk '{print int($2/1000000)-1}'`
export cesarTools=./CESAR2.0/tools
PATH=$PATH:$cesarTools

# create all CESAR jobs
formatGenePred.pl ${inputGenes} ${inputGenes}.forCESAR ${inputGenes}.discardedTranscripts -longest
for transcript in `cut -f1 ${inputGenes}.forCESAR`; do 
   echo "annotateGenesViaCESAR.pl ${transcript} ${alignment} ${inputGenes}.forCESAR ${reference} ${querySpecies} ${outputDir} ${twoBitDir} ${cesarTools} -maxMemory ${maxMemory}"
done > jobList

# realign all genes by executing the jobList or push it to your compute cluster
chmod +x jobList
./jobList
# This will run over night (10.5 h on a single core)

# Convert results into genePred format
for species in `echo $querySpecies | sed 's/,/ /g'`; do 
  echo "bed2GenePred.pl $species $outputDir $resultsDir/$species.gp"
done > jobListGenePred
chmod +x jobListGenePred
./jobListGenePred
# This will take ~15 minutes

# cleanup
rm -rf $outputDir
```
The results are in geneAnnotation/ as one file for each of the 4 query species in genePred file. 

# General workflow of annotating genes in several genomes

## Input data
* a file containing coding genes annotated in the reference genome in UCSC's [genePred format](https://genome.ucsc.edu/FAQ/FAQformat.html#format9). 
The UCSC genome browser provides gene annotations in genePred format for many species. Alternatively, the tools bedToGenePred or gff3ToGenePred can be used to create genePred from bed or gff format. Use UCSC's genePredCheck to check if your input file is valid. 
* a "2bit" directory that contains the genomes of the reference and all query species. 
Each species must have a subdirectory that is identical to the assembly name (e.g. hg38 for human, mm10 for mouse). In this subdirectory, you must have the genome sequence in [2bit format](https://genome.ucsc.edu/goldenpath/help/twoBit.html) and a file called 'chrom.sizes' that contains the size of all scaffolds. This file can be produced by 'twoBitInfo $assembly.2bit chrom.sizes'. 
* a genome alignment in [maf format](https://genome.ucsc.edu/FAQ/FAQformat.html#format5). Index this maf file by running 'mafIndex $alignment.maf $alignment.bb' to create the index in bigBed format.

See the test case above for examples of all input files. 

## Simple Four Step Workflow
### Step 1: Define variables
```
export inputGenes=...   # the genePred file containing the genes in the reference
export reference=...    # the assembly name of the reference
export twoBitDir=...    # the directory containing the genomes and chrom.size files. 
export alignment=...    # the alignment index file
export querySpecies=... # a comma-separated list of the query species that you want to annotate. Each query species must be contained in ${alignment}.
export outputDir=...    # name of CESAR2.0 output directory that will contain exon coordinates (in subdirectories). The directory will be created, if it does not exist. 
export resultsDir=...   # directory containing the final gene annotation (one genePred file per species)
export maxMemory=...    # maximum amount of memory in Gb that CESAR 2.0 can allocate. With 30 Gb, all but 3 human genes succeed. With 50 Gb, all human genes succeed. 
export cesarTools=...   # path to the tools directory. This must contain the 'cesar' binary and the 'extra' subdirectory containing CESAR's profiles and matrices
export PATH=$PATH:$cesarTools
```
It is recommended to add the last export command to your .bashrc.

### Step 2: 
Create the input file for CESAR 2.0 from the genePred gene annotation file by running
```
formatGenePred.pl ${inputGenes} ${inputGenes}.forCESAR ${inputGenes}.discardedTranscripts
```

By default, this step will consider all protein-coding transcripts of each gene. If you only want to consider the longest transcript per gene (instead of all transcripts), use the '-longest' parameter:
```
formatGenePred.pl ${inputGenes} ${inputGenes}.forCESAR ${inputGenes}.discardedTranscripts -longest
```

This step produces an output file that has the coordinates of all coding exons. It discard transcripts with (i) incomplete CDS on either the 3’ or the 5’ end, (ii) with a CDS length that is not a multiple of 3 (e.g. due to programmed frameshift or polymorphisms), and (iii) transcripts with micro introns smaller than 30 bp as such introns are often introduced to sidestep a frameshift. All discarded transcripts are listed in ${inputGenes}.discardedTranscripts.

### Step 3:
Generate the CESAR 2.0 commands for all transcripts by running:
```
for transcript in `cut -f1 ${inputGenes}.forCESAR`; do 
  echo "annotateGenesViaCESAR.pl ${transcript} ${alignment} ${inputGenes}.forCESAR ${reference} ${querySpecies} ${outputDir} ${twoBitDir} ${cesarTools} -maxMemory ${maxMemory}"
done > jobList
```

This creates for every transcript a realignment job for CESAR 2.0. Each line in jobList consists of a single job. Each job is completely independent of any other job, thus each job can be run in parallel to others. 

Execute that jobList file. Either sequentially by doing 
```
chmod +x jobList
./jobList
```
or run it in parallel by using a compute cluster. 

### Step 4: 
After each realignment job succeeded, collect the results as a single genePred file for each query species by running: 
```
for species in `echo $querySpecies | sed 's/,/ /g'`; do 
  echo "bed2GenePred.pl $species $outputDir $resultsDir/$species.gp"
done > jobListGenePred
chmod +x jobListGenePred
./jobListGenePred
```

The final results are in $resultsDir as a single file per query species in genePred format. 

Remove $outputDir if you like. 


# Running CESAR 2.0 directly
## Minimal example

Just call

`./cesar extra/example1.fa`

This will output the re-aligned exon, using the default donor/acceptor profile obtained from human. 
example2/3/4.fa provide further examples.


## Format of the input file
The input file has to be a multi-fasta file. It provides at least one reference and
at least one query sequence. References and queries have to be separated by a
line starting with '#'. References are the exons (together with their reading frame) that you want to align to the query sequence.

Example 1: Aligning a single exon against a single query sequence.
```
>human
gCCTGGGAACTTCACCTACCACATCCCTGTCAGTAGTGGCACCCCACTGCACCTCAGCCTGACTCTGCAGATGaa    
####
>mouse
CCTTTAGGCTTGGCAACTTCACCTACCACATCCCTGTCAGCAGCAGCACACCACTGCACCTCAGCCTGACCCTGCAGATGAAGTGAG
```

The reading frame has to be indicated by lower case letters at the beginning and end of the reference exon. Lower case letters are bases belonging to a codon that is split by the intron. In this example, the 'g' is the third codon base and the first full codon is CCT. The 'aa' at the end are the codon bases 2 and 3 of the split codon. 

Example 2: Aligning a single exon against multiple query sequences
```
>human
GTCACAATCATTGGTTACACCCTGGGGATTCCTGACGTCATCATGGGGATCACCTTCCTGGCTGCTGGGACCAGCGTGCCTGACTGCATGGCCAGCCTCATTGTGGCCAGACAAg
####
>mouse
CTCCAAGGTTACCATCATCGGCTACACACTAGGGATCCCTGATGTCATCATGGGGATCACCTTCCTGGCTGCCGGAACCAGCGTGCCAGACTGCATGGCCAGCCTCATTGTAGCCAGACAAGGTGG
>sheep
TCCCAGGTCACGATCATCGGCTACACGCTGGGGATTCCTGACGTCATCATGGGGAGACAAGGTGGGGCCCACGTGGGGAGGGCTGGGAAGGGAAGCCAGGCCTCCCTACTTAGGGGGTAGGGGGAGCTTGCCTGG
```

Example 3: Gene mode of CESAR 2.0. Provide an input file that lists multiple consecutive or all exons of a gene. By default, CESAR2 assumes that the first given exon is the first coding exon (start codon .. donor), that the last given exon is the last coding exon (acceptor .. stop codon) and that all exons in-between are internal exons (acceptor .. donor). Alternatively, you can specify first/internal/last coding exon by adding the profiles tab-separated after the sequences. If no profiles are specified, CESAR2 outputs a missing profile warning. Reference exons are separated by a line starting with hashes from one or more query sequences.
```
>exon1	extra/tables/human/firstCodon_profile.txt	extra/tables/human/do_profile.txt
AGAGCCAAG
>exon2	extra/tables/human/acc_profile.txt	extra/tables/human/do_profile.txt
TGGAGGAAGAAGCGAATGCGCag
>exon3	extra/tables/human/acc_profile.txt	extra/tables/human/lastCodon_profile.txt
gCTGAAGCGCAAAAGAAGAAAGATGAGGCAGAGGTCCAAGTAA
####
>mouse
GACTCCTGCGCCATGAGAGCGAAGGTGAGCGGCTCTTAGGTGGTGAATCGGGCACCTAGTCCCCGCCATGGTTCCTCTGCGGGCTTCCAGACGGTTTGCCTCGGGTGTTCGCAGTCAGGGAGAGGCTTAAAATTCTTGCTGAAGAAAAGATGGGGTGGGAAAATGAGGGATTCGGCTCTAAAACTGAACCGGTGTCCTTTGTCAAGCCCTGTGTTCTGAGCAGTTTCATGGCCTTGCACAAGCCTGTCTCTAACATTCTTTTTTGTCTCCTCACATGATCGGGTTTTTTTAGTGGCGGAAGAAGAGAATGCGCAGGTACGTTTAATTTTTCAAGACTACCTTGGGGCAGTGGGGGCAAGCTCGGTGTGGGATATTTGGTTGGAAGAAAATATCTGGCGGGAAGGAAGCAGGAGTCGCCGCCCAGTACAGCAGAGCTGGTGCTTGTTAATCTCATCGTCTCTTGTACTCGTGCACTAAGTGTACGTATTGATAGATGTGCAAAGGAAAAAAAAAAAACTCAGGTTTGTGTGCCTTCCATTCCAGGCTGAAGCGCAAGAGAAGAAAGATGAGGCAGAGGTCCAAGTAAGCCAGCCC
```


## Common Parameters

`-f/--firstexon`
Given exon is the first coding exon. Only relevant for single exon mode. The default profile for a start codon is used instead of the acceptor profile.

`-l/--lastexon`
Given exon is the last coding exon. Only relevant for single exon mode. The default profile for stop codons is used instead of the donor profile.

`-m/--matrix <matrix file>`
Use a different codon substitution matrix by specifying a different file.

`-p/--profiles <acceptor> <donor>`
Use different acceptor and donor profiles by specifying different profile files.

`-c/--clade <human|mouse>` (default: `human`)
A shortcut to default sets of substitution matrix and profiles.
For example, `-c human` is synonymous to:
`-m extras/tables/human/eth_matrix.txt -p extras/tables/human/acc_profile.txt extras/tables/human/do_profile.txt`

By default, CESAR2 uses profiles obtained from human.
You can provide profiles for another species in a directory extra/tables/$species and tell CESAR 2.0 to use these profiles by
`./cesar --clade $species example1.fa`

If <clade> contains a slash `/` it will be interpreted as look-up directory for profiles.

**Note:** With `-l` and/or `-f`, the profiles will change accordingly.

`-x/--max-memory`
By default, CESAR2 stops if it is expected to allocate more than 16 GB of RAM.
With this flag, you can set the maximum RAM allowed for CESAR2.
The unit for this parameters is gigabytes (GB). E.g. with `-x 32` you tell CESAR2 to allocate up to 32 GB of RAM.


## Expert Parameters

`-v/--verbosity <n>`
Print extra information to stderr.

n  | Information
------------- | -------------
1  | Input Parameters
2  | List matrices and sequences in memory
3  | Fasta parser and alignment state machine
4  | Emission table initialization and Viterbi path
5  | HMM state creation, transitions and HMM normalization
6  | Full Viterbi step
7  | Initialization and access of emission tables


`-i/--split_codon_emissions <acc split codon emissions> <do split codon emissions>`
Manually define the length of split codons for each reference at once.

**Note:** `-i` is deprecated. Use lower case letters the Fasta file to annotate
split codons and upper case letters for all other codons. Alternatively
separate split codons from full codons with the pipe character `|`.


`-s/--set <name1=value1> .. <nameN=valueN>`
Customize parameters, e.g. transition probabilities. 
If a set of outgoing transitions includes a customized value, CESAR2 will normalized outgoing probabilities to sum to 1.
* `fs_prob` probability of a frame shift, default: `0.0001`
* `ci_prob` codon insertion probability, default: `0.01`
* `ci2_prob` codon insertion continuation, default: `0.2`
* `total_cd_prob` sum of deleting 1..10 codons, default: `0.025`
* `cd_acc` codon deletion at acceptor site, default: `0.012`
* `cd_do` codon deletion at donor site, default: `0.012`
* `c3_i1_do`(codon insertion after codon match near donor site, default: `0.01` via ci_prob
* `i3_i1_do`(codon insertion cont. near donor site, default: `0.4`
* `i3_i1_acc` codon insertion cont. near acceptor site, default: `0.4`
* `no_leading_introns_prob` default: `0.5`
* `no_traling_introns_prob` default: `0.5`
* `intron_del` probabiltiy to skip acc, intron and do between two exons, default: `0.00001` via fs_prob
* `b1_bas` skip the acceptor and the intron, default: `0.0001` via fs_prob
* `b2_bas` skip the acceptor but not the intron: `0.0001` via fs_prob
* `b2_b2` intron continuation, default: `0.9`
* `skip_acc` probabilty to skip acceptors, default: `0.0001` via fs_prob
* `splice_nti` single nucleotide insertion before first codon match, default: `0.0001` via fs_prob
* `nti_nti` single nucleotide insertion continuation, default: `0.25`
* `splice_i1` codon insertion before first codon match, default:  `0.01` via ci_prob
* `i3_i1` codon insertion continuation, default: `0.2` via ci2_prob
* `c3_i1` codon insertion, default: `0.01` via ci_prob
* `bsd_e2` skip donor and intron, default: `0.0001` via fs_prob
* `do2_e2` skip intron at donor site, default:
* `skip_do` probability to skip donors, default: `0.0001` via fs_prob
* `e1_e1` intron continuation at donor site, default: `0.9`
**Note:** In the following values, `%s` will be replaced by the clade name
(e.g. "human"). 
* `eth_file` substitution matrix file, default: `extra/tables/%s/eth_codon_sub.txt`
* `acc_profile` default: `extra/tables/%s/acc_profile.txt`
* `first_codon_profile` default: `extra/tables/%s/firstCodon_profile.txt`
* `u12_acc_profile` default: `extra/tables/%s/u12_acc_profile.txt`
* `do_profile` default: `extra/tables/%s/do_profile.txt`
* `last_codon_profile` default: `extra/tables/%s/lastCodon_profile.txt`
* `u12_donor_profile` default: `extra/tables/%s/u12_donor_profile.txt`


Use with caution!


`-V/--version`
Print the version and exit.


# Credits
CESAR 2.0 was implemented by Peter Schwede (MPI-CBG/MPI-PKS 2017). Virag Sharma (MPI-CBG/MPI-PKS 2017) implemented the workflow to annotate genes in several genomes.

# References
[1] Sharma V, Schwede P, and Hiller M. CESAR 2.0 substantially improves speed and accuracy of comparative gene annotation. Submitted

[2] Sharma V, Elghafari A, and Hiller M. [Coding Exon-Structure Aware Realigner (CESAR) utilizes genome alignments for accurate comparative gene annotation](https://academic.oup.com/nar/article-lookup/doi/10.1093/nar/gkw210). Nucleic Acids Res., 44(11), e103, 2016

