# rmskJoined.sql was originally generated by the autoSql program, which also 
# generated rmskJoined.c and rmskJoined.h.  This creates the database representation of
# an object which can be loaded and saved from RAM in a fairly 
# automatic way.

#RepeatMasker joined annotation record
CREATE TABLE rmskJoinedBaseline (
    bin int unsigned not null,          # bin number for browser speedup
    chrom varchar(255) not null,	# Reference sequence chromosome or scaffold
    chromStart int unsigned not null,	# Start position of feature in chromosome
    chromEnd int unsigned not null,	# End position feature in chromosome
    name varchar(255) not null,	# Short Name of item
    score int unsigned not null,	# Score from 0-1000
    strand char(1) not null,	# + or -
    alignStart int unsigned not null,	# Start position of aligned portion of feature
    alignEnd int unsigned not null,	# End position of aligned portion of feature
    reserved int unsigned not null,	# Used as itemRgb
    blockCount int not null,	# Number of blocks
    blockSizes longblob not null,	# Comma separated list of block sizes
    blockRelStarts longblob not null,	# Start positions rel. to chromStart or -1 for unaligned blocks
    id varchar(255) not null,	# ID to bed used in URL to link back
              #Indices
    INDEX(chrom,bin)
);