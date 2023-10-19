# Run with
# python3 minitrasgo_bot.py

import telebot
import os
import subprocess

correct_password = 'mingo@1234'
bot = telebot.TeleBot("6465612779:AAFZGylWUQTHfqEggF2VTWVNX_c6jOVTZb0")

passwords = {
    '6445882713': 'mingo@1234',
    'user2': 'password2',
    # Add more users and passwords as needed
}


@bot.message_handler(commands=['get_username'])
def start(message):
    user_id = message.from_user.id  # Get the user's unique ID
    user_username = message.from_user.username  # Get the user's username (if available)
    user_first_name = message.from_user.first_name  # Get the user's first name
    user_last_name = message.from_user.last_name  # Get the user's last name (if available)

    # You can now use this user information as needed
    print(f"User ID: {user_id}")
    print(f"Username: {user_username}")
    print(f"First Name: {user_first_name}")
    print(f"Last Name: {user_last_name}")

    # Respond to the user
    bot.reply_to(message, f"Hello, {user_first_name}! Your ID is {user_id}. Your username is {user_username}.")

def require_password(func):
    def wrapper(message):
        chat_id = message.chat.id
        user_id = message.from_user.id

        # Check if the user is in the passwords dictionary
        if str(user_id) in passwords:
            password = passwords[str(user_id)]
            bot.send_message(chat_id, "Please enter your password:")
            bot.register_next_step_handler(message, lambda m: check_password(m, password, func))
        else:
            bot.send_message(chat_id, "You are not authorized to perform this action.")

    return wrapper

def check_password(message, expected_password, func):
    user_id = message.from_user.id
    user_password = message.text

    if user_password == expected_password:
        func(message)
    else:
        bot.send_message(user_id, "Incorrect password. Action canceled.")


# Monitoring --------------------------------------------------------------------

@bot.message_handler(commands=['start'])
def send_welcome(message):
    chat_id = message.chat.id
    bot.reply_to(message, "Howdy, how are you doing? Type anything to see what you can do with the miniTRASGO bot.")

@bot.message_handler(commands=['send_voltage'])
def send_voltage(message):
	binary_path='/home/rpcuser/bin/HV/hv -b 0'
	print(binary_path)

	# Use subprocess to execute the binary file
	result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

	try:
	    # Use subprocess to execute the binary file
	    result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

	    # Check if the execution was successful
	    if result.returncode == 0:
	        print("Execution successful")
	        # Output from the binary is available in result.stdout
	    else:
	        print("Execution failed")
	        # Error message is available in result.stderr
	        print("Error:", result.stderr)
	except Exception as e:
	    print("An error occurred:", e)

	response = f'{result.stdout}'
	print(response)
	bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['send_gas_flow'])
def send_gas_flow(message):
	# /home/rpcuser/bin/flowmeter/bin

	# bash_command = "tail /home/rpcuser/logs/Flow0*"
	# output = os.popen(bash_command).read()
	# bot.send_message(message.chat.id, output)

        binary_path='/home/rpcuser/bin/flowmeter/bin/GetData 0 4'
        print(binary_path)

        # Use subprocess to execute the binary file
        result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        try:
            # Use subprocess to execute the binary file
            result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

            # Check if the execution was successful
            if result.returncode == 0:
                print("Execution successful")
                # Output from the binary is available in result.stdout
            else:
                print("Execution failed")
                # Error message is available in result.stderr
                print("Error:", result.stderr)
        except Exception as e:
            print("An error occurred:", e)

        response = f'{result.stdout}'
        print(response)
        bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['send_internal_environment'])
def send_internal_environment(message):
        bash_command = "tail /home/rpcuser/logs/clean_sensors_bus0*"
        output = os.popen(bash_command).read()
        bot.send_message(message.chat.id, "Date --- T (ºC) --- RH (%) --- P (mbar)")
        bot.send_message(message.chat.id, output)


@bot.message_handler(commands=['send_external_environment'])
def send_external_environment(message):
        bash_command = "tail /home/rpcuser/logs/clean_sensors_bus1*"
        output = os.popen(bash_command).read()
        bot.send_message(message.chat.id, "Date --- T (ºC) --- RH (%) --- P (mbar)")
        bot.send_message(message.chat.id, output)


@bot.message_handler(commands=['send_TRB_rates'])
def send_TRB_rates(message):
        bash_command = "tail /home/rpcuser/logs/clean_rates*"
        output = os.popen(bash_command).read()
        bot.send_message(message.chat.id, "Date --- Asserted - Edge - Accepted --- Multiplexer 1 - M2 - M3 - M4 --- Coincidence Module 1 - CM2 - CM3 - CM4")
        bot.send_message(message.chat.id, output)

# Operation of miniTRASGO ----------------------------------------------------------------

@bot.message_handler(commands=['set_data_control_system'])
@require_password
def set_data_control_system(message):
        bash_command = "sh /media/externalDisk/gate/bin/dcs_om.sh"
        bot.send_message(message.chat.id, 'Started...')
        os.system(bash_command)
        bot.send_message(message.chat.id, 'Executed.')


@bot.message_handler(commands=['start_daq'])
@require_password
def start_daq(message):
        bash_command = "sh /home/rpcuser/trbsoft/userscripts/trb399sc/startup_TRB399.sh"
        bot.send_message(message.chat.id, 'Started...')
        os.system(bash_command)
        bot.send_message(message.chat.id, 'Executed.')


@bot.message_handler(commands=['turn_on_hv'])
@require_password
def turn_on_HV(message):
        binary_path='/home/rpcuser/bin/HV/hv -b 0 -I 1 -V 5.5 -on'
        print(binary_path)

        # Use subprocess to execute the binary file
        result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        try:
            # Use subprocess to execute the binary file
            result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

            # Check if the execution was successful
            if result.returncode == 0:
                print("Execution successful")
                # Output from the binary is available in result.stdout
            else:
                print("Execution failed")
                # Error message is available in result.stderr
                print("Error:", result.stderr)
        except Exception as e:
            print("An error occurred:", e)

        response = f'{result.stdout}'
        print(response)
        bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['turn_off_hv'])
@require_password
def turn_off_HV(message):
        binary_path='/home/rpcuser/bin/HV/hv -b 0 -off'
        print(binary_path)

        # Use subprocess to execute the binary file
        result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

        try:
            # Use subprocess to execute the binary file
            result = subprocess.run(binary_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

            # Check if the execution was successful
            if result.returncode == 0:
                print("Execution successful")
                # Output from the binary is available in result.stdout
            else:
                print("Execution failed")
                # Error message is available in result.stderr
                print("Error:", result.stderr)
        except Exception as e:
            print("An error occurred:", e)

        response = f'{result.stdout}'
        print(response)
        bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['start_run'])
@require_password
def start_run(message):
        bash_command = "sh /home/rpcuser/trbsoft/userscripts/trb399sc/caye_startRun.sh"
        bot.send_message(message.chat.id, 'Started...')
        os.system(bash_command)
        bot.send_message(message.chat.id, 'Finished the run.')


@bot.message_handler(commands=['stop_run'])
@require_password
def stop_run(message):
        bash_command = "kill $(ps -ef | grep '[d]abc' | awk '{print $2}')"
        os.system(bash_command)

# Report and plots ------------------------------------------------------------------------

@bot.message_handler(commands=['send_report'])
def send_report(message):
	with open(f'/media/usb0/gate/system/devices/mingo01/reporting/report_mingo01.pdf', 'rb') as document:
		bot.send_document(message.chat.id, document)

@bot.message_handler(commands=['create_report'])
@require_password
def create_report(message):
        bash_command = "export RPCSYSTEM=sRPC;/home/rpcuser/gate/bin/createReport.sh"
        os.system(bash_command)


# -----------------------------------------------------------------------------------------

@bot.message_handler(func=lambda message: True)
def echo_all(message):
	string = '\
--------------------------------------------\n\
Useful commands for miniTRASGO bot:\n\
--------------------------------------------\n\
\n\
General stuff:\n\
---------------------\n\
- /get_username: show your name, username and ID. Provide this info to csoneira@ucm.es.\n\
--------------------------------------------\n\
Monitoring:\n\
---------------------\n\
- /start: greetings.\n\
- /send_voltage: send a report on the high voltage in the moment.\n\
- /send_gas_flow: send the last values of the gas flow in each RPC.\n\
- /send_internal_environment: send the last values given by the internal environment sensors.\n\
- /send_external_environment: send the last values given by the external environment sensors.\n\
- /send_TRB_rates: send the last rates triggered in the TRB.\n\
--------------------------------------------\n\
\n\
Operation:\n\
---------------------\n\
- /set_data_control_system: script to perform the connection through the I2C protocol from the mingo PC to the hub containing the environment, HV and gas flow sensors.\n\
- /start_daq: initiate the Data Acquisition System so it will be ready to measure. It does not store data.\n\
- /turn_on_hv: self-explanatory.\n\
- /turn_off_hv: self-explanatory, right?\n\
- /start_run: initiate the storing of the data that the daq is receiving.\n\
- /stop_run: kills the run.\n\
--------------------------------------------\n\
\n\
Report and plots:\n\
-----------------------------\n\
- /create_report: creates the pdf report.\n\
- /send_report: send the pdf report generated.\n\
--------------------------------------------\n\
'
	bot.send_message(message.chat.id, string)

bot.infinity_polling()
