# Scan a candidate vcf file, generate window for the variant and mark genotype according to GIAB positives
def VarScan(referenceGenome,bam,Candidate_vcf,Positive_vars,Nprocess):
	if Nprocess<=1:
		RefFile = pysam.FastaFile(referenceGenome)
		SamFile = samfile = pysam.AlignmentFile(bam, "rb")
		fout_training = gzip.open('windows_training.txt.gz','wb')
		fout_validation = gzip.open('windows_validation.txt.gz','wb')
		fout_testing = gzip.open('windows_testing.txt.gz','wb')
		fin = open(Candidate_vcf,'rb')
		for l in fin:
			if l.startswith('##'):
				continue
			elif l.startswith('#'):
				header = l.strip().split('\t')
			else:	
				llist = l.strip().split('\t')
				chrom, pos = llist[0:2]
				if chrom not in ['19','20','21','22','X','Y']:	
					k,p,v = var2kv(l)
					#region = Region.Region(ref,samfile, chrom, int(pos))
					if k in Positive_vars:
						GT = get_Genotype(llist)
						region = Region.CreateRegion(RefFile, SamFile, chrom, pos, str(GT)) #Create a Region according to a site
					else:
						region = Region.CreateRegion(RefFile, SamFile, chrom, pos, '0') 
					#Pulse(region)
					fout_training.write(region.write()+'\n')
				elif chrom in ['19']:
					k,p,v = var2kv(l)
					if k in Positive_vars:
						GT = get_Genotype(llist)
						region = Region.CreateRegion(RefFile, SamFile, chrom, pos, str(GT))
					else:
						region = Region.CreateRegion(RefFile, SamFile, chrom, pos, '0') 
					fout_validation.write(region.write()+'\n')
				elif chrom in ['20','21','22']:
					k,p,v = var2kv(l)
					#region = Region.Region(ref,samfile, chrom, int(pos))
					if k in Positive_vars:
						GT = get_Genotype(llist)
						region = Region.CreateRegion(RefFile, SamFile, chrom, pos, str(GT)) #Create a Region according to a site
					else:
						region = Region.CreateRegion(RefFile, SamFile, chrom, pos, '0') 
					#Pulse(region)
					fout_testing.write(region.write()+'\n')
		fout_training.close()
		fout_testing.close()

	else:
		jobs = []
		for i in range(Nprocess):
			p = multiprocessing.Process(target=worker, args=(i,))
			jobs.append(p)
			p.start()