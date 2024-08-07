import asyncio

async def communicate_with_server():
	reader, writer = await asyncio.open_connection('127.0.0.1', 8000)
	print("Connected to server")

	code, last = "", ""
	exit_var = False
	while not exit_var:
		val = input("> ")
		if val == "$$$":
			exit_var = True
		else:
			if val == "" and last == "":
				message = code + "$x$"
				print("Sending message")
				writer.write(message.encode())
				await writer.drain()

				response = await reader.read(1000)
				print(f'Received: {response.decode()}')
				code = ""
				last = ""
			else:
				code += val + "\n"
				last = val
	print("Closing the connection")
	writer.close()
	await writer.wait_closed()
asyncio.run(communicate_with_server())
