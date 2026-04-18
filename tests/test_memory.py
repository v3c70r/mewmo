"""Basic tests for the memory system."""

import tempfile
from mewmo import Memory


def test_store_and_recall():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        mem = Memory(data_dir=tmp)
        try:
            id1 = mem.store("The cat sat on the mat", {"topic": "animals"})
            id2 = mem.store("Python is a programming language", {"topic": "tech"})
            id3 = mem.store("Neural networks learn from data", {"topic": "tech"})

            assert mem.count() == 3

            results = mem.recall("machine learning")
            assert len(results) > 0
            ids = [r["id"] for r in results]
            assert id3 in ids

            results = mem.recall("pets and animals")
            assert results[0]["id"] == id1
        finally:
            mem.close()


def test_forget():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        mem = Memory(data_dir=tmp)
        try:
            file_id = mem.store("temporary memory")
            assert mem.count() == 1
            mem.forget(file_id)
            assert mem.count() == 0
        finally:
            mem.close()


def test_deduplication():
    """Storing the same content twice should not create duplicates."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        mem = Memory(data_dir=tmp)
        try:
            id1 = mem.store("same content")
            id2 = mem.store("same content")
            assert id1 == id2
            assert mem.count() == 1
        finally:
            mem.close()


if __name__ == "__main__":
    test_store_and_recall()
    test_forget()
    test_deduplication()
    print("All tests passed!")
