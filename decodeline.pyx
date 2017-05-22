# Used For Training, only read tensor and label
# Only Norm Qual, not bases and strand
def DecodeRecord(line, WIDTH, HEIGHT):
    chrom, start, end, ref, alt, label, window = line.strip().split('\t')
    Alignment = window[ 0 : WIDTH * (HEIGHT) ]
    Qual = window[ WIDTH * (HEIGHT) : WIDTH * (HEIGHT)*2]
    Strand = window[ WIDTH * (HEIGHT)*2 : WIDTH * (HEIGHT)*3]
    p1 = [((float(x)-3) / 3)  for x in Alignment]
    p2 = [((float(ord(x) - 33) -30) / 30) for x in Qual]
    p3 = [float(x)-1 for x in Strand]
    return p1 + p2 + p3, label

# Used For Calling, also read other info beside tensor and label
def DecodeRecord_WithInfo(line, WIDTH, HEIGHT):
    chrom, start, end, ref, alt, label, window = line.strip().split('\t')
    Alignment = window[ 0 : WIDTH * (HEIGHT) ]
    Qual = window[ WIDTH * (HEIGHT) : WIDTH * (HEIGHT)*2]
    Strand = window[ WIDTH * (HEIGHT)*2 : WIDTH * (HEIGHT)*3]
    p1 = [((float(x)-3) / 3)  for x in Alignment]
    p2 = [((float(ord(x) - 33) -30) / 30) for x in Qual]
    p3 = [float(x)-1 for x in Strand]
    return p1 + p2 + p3, chrom, start, ref, alt, label