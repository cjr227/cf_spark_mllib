from pyspark.ml.recommendation import *

# Prepare the data
rawUserArtistData = sc.textFile("hdfs:///user/ds/user_artist_data.txt")
rawArtistData = sc.textFile("hdfs:///user/ds/artist_data.txt")

def not_tab(x):
    return x != '\t'

def span(input_list,predicate):
    prefix = []
    rest = []
    for item in input_list:
        if predicate(item):
            prefix.append(item)
        else:
            break
    rest = [x for x in input_list if x not in prefix]
    return (prefix,rest)

assert span('123\t456',predicate=not_tab) == (['1',2','3'],['\t','4','5,','6'])

def artist_id_map(line):
    (id, name) = span(line,predicate=not_tab)
	if name in ['',' ',None]:
	    return None
	else:
		try:
			return (int(id),name.strip())
		except ValueError,e:
			return None

artistByID = rawArtistData.flatMap(artist_id_map)

rawArtistAlias = sc.textFile("hdfs:///user/ds/artist_alias.txt")

def alias_map(line):
    tokens = line.split('\t')
	if tokens == []:
		return None
	if tokens[0] in ['',' ',None]:
		return None
	else:
		return (int(tokens[0]),int(tokens[1]))

artistAlias = rawArtistAlias.flatMap(alias_map).collectAsMap()

# Building a First Model
bArtistAlias = sc.broadcast(artistAlias)

def prepare_training_data(line):
    userID, artistID, count = map(int,line.split(' '))
	  finalArtistID = bArtistAlias.get(artistID,artistID)
    return Rating(userID, finalArtistID, count)
 
 trainData = rawUserArtistData.map(prepare_training_data).cache()
