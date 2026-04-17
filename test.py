from DocumentExtraction import Extractedfiles
from procecssor import InvertedIndex
from Queries import Queries


def main():


    files = Extractedfiles()
    files.readData()
    docs = files.getfiles()

    print("Total Documents:", len(docs))
    print("\nSample Document:\n")
    # print(docs[0][:500]) 

    print("\nBuilding Inverted Index...\n")

    processor = InvertedIndex()
    processor.documentProcessing()

    print("Index built successfully!")
    print("Vocabulary size:", len(processor.words))

    # processor.writeToFile()


    query_engine = Queries(processor)

    test_queries = [
        " massive inflow of refugees",
    ]

    print("\n==============================")
    print("TESTING VSM QUERY SYSTEM")
    print("==============================\n")

    for q in test_queries:
        print(f"\n🔍 Query: {q}")

        results = query_engine.process_query(q)

        if not results:
            print("❌ No results found")
        else:
            print("📄 Top Results:")
            for doc_id, score in results[:5]:
                print(f"Doc {doc_id} | Score: {round(score, 4)}")


if __name__ == "__main__":
    main()