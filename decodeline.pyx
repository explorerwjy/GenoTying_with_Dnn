def DecodeRecord(line, WIDTH, HEIGHT):
    chrom, start, end, ref, alt, label, window = line.strip().split('\t')
    Alignment = window[ 0 : WIDTH * (HEIGHT+1) ]
    Qual = window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
    Strand = window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
    p1 = [float(x) for x in Alignment]
    p2 = [((float(ord(x) - 33) / 60) - 0.5) for x in Qual]
    p3 = [float(x) for x in Strand]
    return p1 + p2 + p3, label

def DecodeRecord2(line, WIDTH, HEIGHT):
    chrom, start, end, ref, alt, label, window = line.strip().split('\t')
    Alignment = window[ 0 : WIDTH * (HEIGHT+1) ]
    Qual = window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
    Strand = window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
    p1 = [float(x) for x in Alignment]
    p2 = [((float(ord(x) - 33) / 60) - 0.5) for x in Qual]
    p3 = [float(x) for x in Strand]
    return p1 + p2 + p3, chrom, start, ref, alt

def DecodeRecord3(line, WIDTH, HEIGHT):
    chrom, start, end, ref, alt, label, window = line.strip().split('\t')
    Alignment = window[ 0 : WIDTH * (HEIGHT+1) ]
    Qual = window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
    Strand = window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
    p1 = [float(x) for x in Alignment]
    p2 = [((float(ord(x) - 33) / 60) - 0.5) for x in Qual]
    p3 = [float(x) for x in Strand]
    return p1 + p2 + p3, chrom, start, ref, alt, label

def DecodeLine(window, WIDTH, HEIGHT):
    #chrom, start, end, ref, alt, label, window = line.strip().split('\t')
    Alignment = window[ 0 : WIDTH * (HEIGHT+1) ]
    Qual = window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
    Strand = window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
    p1 = [float(x) for x in Alignment]
    p2 = [((float(ord(x) - 33) / 60) - 0.5) for x in Qual]
    p3 = [float(x) for x in Strand]
    return p1 + p2 + p3

def DecodeLine2(window, WIDTH, HEIGHT):
    #chrom, start, end, ref, alt, label, window = line.strip().split('\t')
    Alignment = window[ 0 : WIDTH * (HEIGHT+1) ]
    Qual = window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
    Strand = window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
    p1 = [float(x)/6 - 0.5 for x in Alignment]
    p2 = [((float(ord(x) - 33) / 60) - 0.5) for x in Qual]
    p3 = [float(x)/2 - 0.5 for x in Strand]
    return p1 + p2 + p3

def _DecodeRecord3(line, WIDTH, HEIGHT):
    chrom, start, end, ref, alt, label, window = line.strip().split('\t')
    Alignment = window[ 0 : WIDTH * (HEIGHT+1) ]
    Qual = window[ WIDTH * (HEIGHT+1) : WIDTH * (HEIGHT+1)*2]
    Strand = window[ WIDTH * (HEIGHT+1)*2 : WIDTH * (HEIGHT+1)*3]
    p1 = [float(x)/6-0.5 for x in Alignment]
    p2 = [((float(ord(x) - 33) / 60) - 0.5) for x in Qual]
    p3 = [float(x)/2-0.5 for x in Strand]
    return p1 + p2 + p3, chrom, start, ref, alt, label