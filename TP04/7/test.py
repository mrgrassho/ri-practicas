from index import IndexIR

i = IndexIR(skip_pointers=True)
for e in range(1800):
    i.add_entry('def',e)
i.indexing_ready()
print(i.get_plist('def'))