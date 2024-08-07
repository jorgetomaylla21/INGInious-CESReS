from fastapi import FastAPI, HTTPException, Query
#import httpx

app = FastAPI()

VM_SERVER_URL = "http://127.0.0.1:8000"

import asyncio

async def cesres_server(message):
        reader, writer = await asyncio.open_connection('130.104.228.136', 8000)
        #message = "while True:/n/tprint('hi')"
        message += "$x$"
        writer.write(message.encode())
        await writer.drain()

        response = await reader.read(1000)
        #output = response.decode()
        print(f'Received: {response.decode()}')
        
        writer.close()
        await writer.wait_closed()
        return response.decode()

def get_max_label(labels):
    # the model responds with labels in csv format => label:number; repeat
    max_label = None
    max_probability = 0
    for parsed_label in labels.split(";"):
        [label, num] = parsed_label.split(":")
        #new_labels.append((label, round(float(num), 2)))
        if float(num) > max_probability:
            max_label = label, round(float(num), 2)
            max_probability = float(num)
    return max_label

def get_label_mapping(label):
    #dictionary containing static mapping for each error label
    mapping = {"miss_return":"Check your return statement position and return value", 
               "print_return":"Make sure you have not mixed up print and return statements",
               "bad_division":"Check for the usage of the '//' integer division operator, or the '/' float division operator",
               "bad_variable":"Check your variable declarations and make sure they are within scope",
               "bad_loop":"Make sure your while conditions terminate",
               "bad_range":"Check the range of your loop. It is common to overshoot the range by 1, so make sure the limits of your range do not cause undesired behavior",
               "bad_assign":"Make sure to initialize your variables correctly. A common mistake is to use the comparison operator '==' instead of the assignment operator '='",
               "bad_list":"Make sure you are handling the correct list. It is helpful to know the list type and content",
               "bad_file":"Make sure your filename is correct and can be read",
               "miss_try":"Check your try and catch blocks",
               "miss_parenthesis":"You might have a missing set of (). Make sure all '(' have a corresponding ')'",
               "hardcoded_arg":"Make sure your output is dependent on the input, and not a hard coded value",
               "overwrite_not_increment":"",
               "miss_loop":"If you have an iterable object (like a list), make sure you use a loop to check/manipulate its content",
               "failed":"Try debugging with print statements to find the error source",
               "correct":"No obvious errors found. If this input is still incorrect, try reading the test case and understand the input that your code fails"
            }
    return mapping[label]


@app.get("/data")
async def get_data_from_server(message: str = Query(..., min_length=1)):
    try:
        response = await cesres_server(message)
        label, num = get_max_label(response)
        return {"response":f"From AI Assistant: Your code has an error in the form of {label} with probability {str(num)}.", "tips":f"Recommendations: {get_label_mapping(label)}."}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with VM server: {e}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
