import paramiko
import os


class Server():
	def __init__(self, ip, port, username, pkey, timeout = 30, print_debug_info = True):
		self.ip = ip
		self.port = port
		self.username = username
		self.pkey = pkey
		self.timeout = timeout
		self.print_debug_info = print_debug_info

		self.ssh = None
		self.sftp = None
		
	def connect(self):
		self.ssh = paramiko.SSHClient()
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		try:
			# ssh connect
			self.ssh.connect(self.ip, username = self.username, pkey = self.pkey, timeout = self.timeout)

			# sftp connect
			t = paramiko.Transport(sock = (self.ip, self.port))
			t.connect(username = self.username, pkey = self.pkey)
			self.sftp = paramiko.SFTPClient.from_transport(t)
			
			print ('Successfully connect to %s' % (self.ip))
			return 0
		except:
			print ('[Err] Failed to connect %s' % (self.ip))
			return -1

	def close(self):
		self.ssh.close()
		self.sftp.close()
		print ('Close the connection from %s' % (self.ip))

	def exec_cmd(self, cmd):
		if self.ssh is None:
			if self.print_debug_info:
				print ('[Err] Failed to execute command')
			return -1

		stdin, stdout, stderr = self.ssh.exec_command(cmd)
		out_info = stdout.read()
		err_info = stderr.read()
		if len(err_info) > 0:
			if self.print_debug_info:
				print ('[Err] %s' % (err_info.strip()))
			return -1
		if len(out_info) > 0:
			return out_info

	def sftp_put(self, local_file, server_file, send_flag = True):
		try:
			# Check server directory exist or not
			server_dir_path,_ = os.path.split(server_file)
			result = self.sftp_checkdir(server_dir_path)
			if result == -1:
				return -1

			self.sftp.put(local_file,server_file)
			
			if send_flag:
				# send a flag file to server
				local_file_path, local_file_name = os.path.split(local_file) 
				local_file_name, _ = os.path.splitext(local_file_name)
				local_flag_path = os.path.join(local_file_path, local_file_name)

				server_file_path, server_file_name = os.path.split(server_file) 
				server_file_name, _ = os.path.splitext(server_file_name)
				server_flag_path = os.path.join(server_file_path, server_file_name).replace('\\', '/')  # replace '\' on windows to '/' on linux file system

				local_flag = open(local_flag_path, 'w')
				local_flag.close()
				# print (local_flag_path)
				# print (server_flag_path)
				self.sftp.put(local_flag_path,server_flag_path)
				os.remove(local_flag_path)			# remove local flag file 

			if self.print_debug_info:
				print ('\t[SFTP] Successfully send %s [=====>] %s' % (local_file, server_file))
			return 0
		except:
			if self.print_debug_info:
				print ('\t[Err] Failed to send %s [=====>] %s' % (local_file, server_file))
			return -1

	def sftp_get(self, server_file, local_file):
		try:
			# Check local directory exist or not
			local_dir_path,_ = os.path.split(local_file)
			if not os.path.exists(local_dir_path):
				os.mkdir(local_dir_path)

			self.sftp.get(server_file,local_file)
			if self.print_debug_info:
				print ('\t[SFTP] Successfully get %s [<=====] %s' % (local_file, server_file))
			return 0
		except:
			if self.print_debug_info:
				print ('\t[Err] Failed to get %s [<=====] %s' % (local_file, server_file))
			return -1

	def sftp_checkdir(self, dir_path):
		# Check directory exist or not. If not, create the directory.
		try:
			self.sftp.listdir_attr(dir_path)
			if self.print_debug_info:
				print ('\t[SFTP] Exist directory %s on server' % (dir_path))
			return 0
		except:
			result = self.sftp.mkdir(dir_path) 
			if result is None:
				if self.print_debug_info:
					print ('\t[SFTP] Create directory %s on server' % (dir_path))
				return 0
			else:
				if self.print_debug_info:
					print ('\t[Err] Failed to create directory %s' % (dir_path))
				return -1
		



